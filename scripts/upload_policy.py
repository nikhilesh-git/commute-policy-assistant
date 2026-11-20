"""Upload a policy file as a new version, through the real API (admin JWT).

Usage (inside the container or with the venv):
  python scripts/upload_policy.py policies/nimbus/transport-policy.v2.md --user admin@nimbus

The company is NOT a parameter: it comes from the admin's token.
"""

import argparse
import re
import sys
from pathlib import Path

import httpx

TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--user", required=True, help="admin username, e.g. admin@nimbus")
    p.add_argument("--password", default="demo1234")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--slug", help="defaults to the file name without .vN.md")
    args = p.parse_args()

    content = args.file.read_text(encoding="utf-8")
    slug = args.slug or re.sub(r"\.v\d+\.md$", "", args.file.name)
    m = TITLE.search(content)
    title = m.group(1) if m else slug

    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        r = client.post("/auth/token", json={"username": args.user, "password": args.password})
        r.raise_for_status()
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        r = client.post(f"/admin/policies/{slug}/versions", json={"title": title, "content": content}, headers=headers)
        print(r.status_code, r.text)
        return 0 if r.status_code == 201 else 1


if __name__ == "__main__":
    sys.exit(main())
