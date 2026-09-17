import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.pipelines.*
import jetbrains.buildServer.configs.kotlin.triggers.vcs

version = "2026.1"

private val pipelineId = "Opake_LinuxAmd64Tests"

private data class TestShard(
    val id: String,
    val label: String,
    val path: String,
)

private fun jobId(shard: TestShard) = "LinuxAmd64_${shard.id}"

private val linuxAmd64TestShards = listOf(
    TestShard("Accounting", "opake-accounting", "packages/opake-accounting"),
    TestShard("Alignment", "opake-alignment", "packages/opake-alignment"),
    TestShard("Auditing", "opake-auditing", "packages/opake-auditing"),
    TestShard("Base", "opake-base", "packages/opake-base"),
    TestShard("Dpftrl", "opake-dpftrl", "packages/opake-dpftrl"),
    TestShard("Dpsgd", "opake-dpsgd", "packages/opake-dpsgd"),
    TestShard("Engine", "opake-engine", "packages/opake-engine"),
    TestShard("Optimizers", "opake-optimizers", "packages/opake-optimizers"),
    TestShard("Patches", "opake-patches", "packages/opake-patches"),
    TestShard("Transformers", "opake-transformers", "packages/opake-transformers"),
    TestShard("Integration", "integration", "tests"),
)

private val setupScript = """
    set -euo pipefail

    "${'$'}OPAKE_PYTHON" --version
    rustc --version
    uv --version

    uv venv --python "${'$'}OPAKE_PYTHON"
    uv sync --locked --group dev --all-packages --extra all
"""

private fun testScript(shard: TestShard) = """
    set -euo pipefail

    coverage_report="coverage-linux-amd64-${shard.id.lowercase()}.xml"
    junit_report="junit-linux-amd64-${shard.id.lowercase()}.xml"

    set +e
    timeout --preserve-status 30m \
        uv run --no-sync pytest ${shard.path} \
        -m "not cuda and not mps and not slow" \
        -n auto --dist loadscope \
        --cov=opake \
        --cov-report=xml:"${'$'}coverage_report" \
        --junitxml="${'$'}junit_report" \
        --durations=0 --durations-min=5 \
        -q
    status=${'$'}?
    set -e

    exit "${'$'}status"
"""

project {
    pipeline {
        id(pipelineId)
        name = "Opake Linux amd64 tests"

        repositories {
            repository(DslContext.settingsRoot)
        }

        triggers {
            vcs {
                branchFilter = "+:*"
            }
        }

        linuxAmd64TestShards.forEach { shard ->
            job {
                id(jobId(shard))
                name = shard.label

                params {
                    param("env.OPAKE_PYTHON", "python3.11")
                }

                requirements {
                    equals("teamcity.agent.jbHosted", "true")
                    startsWith("system.agent.name", "Linux-Large")
                }

                features {
                    xmlReport {
                        id = "JUnitResults"
                        reportType = XmlReport.XmlReportType.JUNIT
                        rules = "+:junit-linux-amd64-${shard.id.lowercase()}.xml"
                    }
                }

                steps {
                    script {
                        id = "SetupTestEnvironment"
                        name = "Set up test environment"
                        scriptContent = setupScript
                    }
                    script {
                        id = "RunTests"
                        name = "Run tests"
                        scriptContent = testScript(shard)
                    }
                }

                outputFiles {
                    pipelineArtifacts("coverage-linux-amd64-${shard.id.lowercase()}.xml")
                    pipelineArtifacts("junit-linux-amd64-${shard.id.lowercase()}.xml")
                }
            }
        }
    }
}
