import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.7.20"
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.4.0")
    implementation("com.londogard:nlp:1.2.0")
    implementation("org.jetbrains.kotlinx:dataframe:0.8.1")
    implementation("org.jetbrains.kotlinx:multik-core:0.2.1")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.1")
    implementation("ai.djl.pytorch:pytorch-engine:0.19.0")
    implementation("ai.djl.onnxruntime:onnxruntime-engine:0.19.0")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

application {
    mainClass.set("MainKt")
}