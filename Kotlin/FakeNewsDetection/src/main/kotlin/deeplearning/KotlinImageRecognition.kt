package deeplearning

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import java.io.File

/**
 * Building and training model of clothes recognition
 */
fun main() {
    // building a model
    val model = Sequential.of(
        Input(28,28,1),
        Flatten(),
        Dense(300),
        Dense(100),
        Dense(10)
    )

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.printSummary()

        // load dataset
        val (train, test) = fashionMnist()

        // train
        it.fit(
            dataset = train,
            epochs = 10,
            batchSize = 100
        )

        val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")

        // saving
        it.save(File("model/my_model"), writingMode = WritingMode.OVERRIDE)
    }
}