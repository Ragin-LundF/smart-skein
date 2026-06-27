plugins {
    `java-platform`
    id("skein.published-library-conventions")
}

description = "Skein Bill of Materials — aligns all Skein module versions for consumers."

extra["publishName"] = "Skein BOM"
extra["publishDescription"] = "Bill of Materials aligning all Skein module versions."

dependencies {
    constraints {
        // Published Skein modules are added here as each module is created.
        api("io.github.ragin-lundf:skein-text:${project.version}")
        api("io.github.ragin-lundf:skein-classify:${project.version}")
        api("io.github.ragin-lundf:skein-extract:${project.version}")
        api("io.github.ragin-lundf:skein-store-postgres:${project.version}")
        api("io.github.ragin-lundf:skein-cli:${project.version}")
    }
}
