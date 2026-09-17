# Security Policy

## Supported Versions

Security fixes are made for the latest released version of Opake and the
current `main` branch. Reports affecting older releases are assessed case by
case; please upgrade before reporting when possible.

## Reporting a Vulnerability

Report suspected vulnerabilities privately to
[security@jetbrains.com](mailto:security@jetbrains.com). If your report
contains sensitive details, encrypt it with the [JetBrains Security PGP
key][pgp-key]. Do not open a public issue or discussion for a suspected
vulnerability.

Please include:

- a clear description of the issue and its potential impact;
- steps or a minimal reproduction that demonstrate the behavior;
- affected versions, commits, platforms, and dependencies;
- any suggested mitigations or fixes, if known.

Do not include exploit details or proof-of-concept code in public issues,
discussions, pull requests, or commits before maintainers have triaged a
security report.

## What Is Not a Security Report

Opake is a research software library. Report routine library defects,
including non-sensitive concerns about differential privacy guarantees, privacy
accounting, mechanisms, clipping, noise, and sampling, through the [public
issue tracker][issues]. Use the private channel when a defect may have a
material privacy impact, enables a security vulnerability, exposes
confidential information, or otherwise requires coordinated disclosure.

## Handling and Disclosure

Reports are handled under JetBrains' [Coordinated Disclosure
Policy][coordinated-disclosure], including its acknowledgement, remediation,
and disclosure process. Please allow reasonable time for a fix or mitigation
before public disclosure. When a fix is released, maintainers will publish
affected versions, remediation guidance, and credit where requested.

This policy covers vulnerabilities in the repository and its released packages,
including dependency vulnerabilities that materially affect Opake. Report
dependency vulnerabilities to their upstream project as well.

[coordinated-disclosure]: https://www.jetbrains.com/legal/docs/terms/coordinated-disclosure/
[issues]: https://github.com/JetBrains-Research/opake/issues
[pgp-key]: https://www.jetbrains.com/privacy-security/
