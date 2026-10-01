# Security and safety

## Reporting a vulnerability

Please report security vulnerabilities privately through GitHub's "Report a vulnerability" button on the repository's Security tab, not as a public issue. Reports are acknowledged as soon as possible and fixed before public disclosure.

Relevant examples: ways to make the API execute code or read files, ways to bypass the candidate validation so a judge's answer reaches the output without being offered, and denial-of-service through crafted input.

## Clinical safety issues

A systematic mapping error is a safety issue even when it's not a security vulnerability. Examples: a negation that's lost, a frequent phrase that maps confidently to the wrong concept, an abbreviation resolved without enough evidence. Report it with the "Wrong mapping" issue template, de-identified. If it could cause harm in a live deployment, use the private vulnerability report instead.

## Deployment notes

- The API has **no authentication**. The Docker setup binds it to `127.0.0.1`. Put it behind your organization's API gateway or reverse proxy before exposing it on a network.
- Clif sends no data to external services. The optional judges (Laya, a local LLM) and embedding services are configured by URL. Make sure those URLs point inside your network.
- Clinical text may appear in debug output and server logs. Treat both as clinical data, and disable debug output for routine use.
- Clif doesn't diagnose and isn't a certified medical device. Use it where codes are reviewed, and validate it on your own documentation first.
