pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

rootProject.name = "smart-skein"

include("skein-bom")
include("skein-text")
include("skein-classify")
include("skein-extract")
include("skein-store-postgres")
include("skein-cli")
include("examples")
