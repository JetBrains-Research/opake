# TeamCity pipeline prototype

This directory defines the TeamCity 2026.1 Kotlin DSL pipeline that replaces
the GitHub Actions Linux amd64 test matrix. Kotlin expands the shard list in
`settings.kts` into one pipeline job per package plus the integration suite
when TeamCity imports versioned settings.

Set `teamcity.server.url` in `pom.xml` to the TeamCity server that will import
these settings, then enable versioned settings for the Opake TeamCity project
with this repository as the settings VCS root and `.teamcity` as its settings
path. The same VCS root must track the branches and pull requests that should
run the pipeline; the pipeline uses `DslContext.settingsRoot` as its main
repository and triggers on every tracked branch.

Jobs run on TeamCity's built-in `Linux-Large` hosted agents. Their image must
provide `python3.11`, Rust stable, GNU `timeout`, and `uv`. Each job creates its
own Python 3.11 environment, runs the same locked dependency installation and
pytest command as the GitHub Actions Linux amd64 lane, publishes JUnit and
coverage XML artifacts, and imports JUnit results with TeamCity's XML Report
Processing feature.

To validate the generated TeamCity configuration locally:

```bash
cd .teamcity
mvn teamcity-configs:generate
```
