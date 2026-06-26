// A published Skein library: the common baseline plus Dokka API docs and Maven/JReleaser publishing.
// The 'examples' module uses skein.common-conventions instead, since it is never published.
plugins {
    id("skein.common-conventions")
    id("skein.dokka-conventions")
    id("skein.publishing-conventions")
}
