"""BM25 keyword ranking and hybrid retrieval."""

import pytest

from app.ingestion import ingest_policy_version
from app.keyword_search import BM25Index, tokenize
from app.retrieval import active_version_ids, retrieve
from tests.conftest import read_policy
from tests.test_retrieval import SWEEP_QUESTIONS, make_corpus

# --- tokenizer and BM25 maths ------------------------------------------------------


def test_tokenize_lowercases_drops_stopwords_and_folds_plurals():
    assert tokenize("Can my Interview Candidates use the cabs?") == ["interview", "candidate", "use", "cab"]
    assert tokenize("policies pass") == ["policy", "pass"]  # 'ies' -> 'y'; 'ss' is not a plural


def test_rare_matching_word_ranks_first():
    index = BM25Index([
        "Drops are provided for shifts that end between 6 PM and 2 AM.",
        "Visitors and interview candidates may use company transport.",
        "Escort after 8 PM for women employees on every drop.",
    ])
    assert index.top("Can my interview candidate use the cab?", 3)[0] == 1


def test_rare_terms_weigh_more_than_common_ones():
    index = BM25Index(["drop escort", "drop guest", "drop airport"])
    assert index.idf["airport"] > index.idf["drop"]


def test_documents_without_any_match_are_not_returned():
    index = BM25Index(["escort after 8 pm", "driver waits 5 minutes"])
    assert index.top("airport", 5) == []
    assert index.top("", 5) == []


# --- hybrid retrieval ------------------------------------------------------------------

@pytest.fixture(scope="module")
def hybrid(tmp_path_factory):
    conn, store, settings = make_corpus(tmp_path_factory.mktemp("hybrid"), retrieval_mode="hybrid")
    yield conn, store, settings
    conn.close()


def sections(result):
    return [c.section_id for c in result.chunks]


@pytest.mark.parametrize("company, question, section", [
    ("nimbus", "Will a guard come with me on a late night drop?", "4.1"),
    ("nimbus", "Who approves an airport trip?", "6.2"),
    ("orion", "How much is deducted for missing my cab?", "8.2"),
    ("kestrel", "Is there a limit on the fare of an ad-hoc trip?", "6.2"),
    ("nimbus", "How many hours before my shift must I book a cab?", "3.2"),  # dense ranks this 5th
])
def test_hybrid_sends_the_relevant_section_to_the_llm(hybrid, company, question, section):
    conn, store, settings = hybrid
    assert section in sections(retrieve(conn, store, settings, company, question))


def test_hybrid_finds_the_section_dense_misses(hybrid):
    conn, store, settings = hybrid
    q = "Can my interview candidate use the office cab?"
    dense = retrieve(conn, store, settings.model_copy(update={"retrieval_mode": "dense"}), "nimbus", q)
    assert "7" not in sections(dense)  # the baseline failure this improvement targets
    assert "7" in sections(retrieve(conn, store, settings, "nimbus", q))


def test_hybrid_isolation_sweep(hybrid):
    conn, store, settings = hybrid
    for company in ("nimbus", "orion", "kestrel"):
        active = set(active_version_ids(conn, company))
        for q in SWEEP_QUESTIONS:
            result = retrieve(conn, store, settings, company, q)
            assert result.dropped == 0
            assert len(result.chunks) <= settings.max_citations
            for chunk in result.candidates:
                assert chunk.company_id == company and chunk.version_id in active


def test_hybrid_still_gates_off_topic_questions(hybrid):
    conn, store, settings = hybrid
    assert retrieve(conn, store, settings, "nimbus", "What is the capital of France?").chunks == []


def test_keyword_index_follows_the_active_version(tmp_path):
    """The BM25 cache is keyed by version ids, so a new version is picked up at once."""
    conn, store, settings = make_corpus(tmp_path, retrieval_mode="hybrid")
    q = "escort for women employees on drops"
    before = retrieve(conn, store, settings, "nimbus", q)
    assert any("8:00 PM" in c.text for c in before.chunks)

    v2 = ingest_policy_version(conn, store, settings, "nimbus", "transport-policy",
                               "Nimbus Tech Employee Transport Policy", read_policy("nimbus/transport-policy.v2.md"))
    after = retrieve(conn, store, settings, "nimbus", q)
    assert {c.version_id for c in after.candidates} == {v2.version_id}
    assert any("7:30 PM" in c.text for c in after.chunks)
    assert not any("8:00 PM" in c.text for c in after.candidates)
