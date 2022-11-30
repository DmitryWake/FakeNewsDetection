import com.londogard.nlp.meachinelearning.predictors.asAutoOneHotClassifier
import com.londogard.nlp.meachinelearning.predictors.classifiers.LogisticRegression
import com.londogard.nlp.meachinelearning.predictors.classifiers.NaiveBayes
import com.londogard.nlp.meachinelearning.vectorizer.TfIdfVectorizer
import com.londogard.nlp.tokenizer.SimpleTokenizer
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import org.jetbrains.kotlinx.dataframe.io.readTSV
import org.jetbrains.kotlinx.dataframe.size

/**
 * Testing ML algorithms for fake news detection
 */
fun main() {
    // dataset
    val trainDf = DataFrame.readTSV("after_process.tsv").dropNA()
    println(trainDf.head())

    // split into tokens
    val tokenizer = SimpleTokenizer()
    val dfWithTokens = trainDf.add {
        "tokens" from trainDf["title"].map { row -> tokenizer.split(row.toString()) }
    }
    println(dfWithTokens.head())

    // train data
    val trainSize = (dfWithTokens.size().nrow * 0.8).toInt()
    val testSize = (dfWithTokens.size().nrow * 0.2).toInt()

    val train = dfWithTokens.take(trainSize)
    val test = dfWithTokens.takeLast(testSize)
    val xTrain = train["tokens"]
    val xValid = test["tokens"]
    val yTrain = train["is_fake"].convertToInt()
    val yTest = test["is_fake"].convertToInt()

    // vectorization
    val vectorizer = TfIdfVectorizer<Int>()
    val xTrainVec = vectorizer.fitTransform(xTrain.toList() as List<List<String>>)
    val xValidVec = vectorizer.transform(xValid.toList() as List<List<String>>)

    // naive bayes
    val naiveBayesClassifier = NaiveBayes().asAutoOneHotClassifier<NaiveBayes, Int>()
    naiveBayesClassifier.fit(xTrainVec, yTrain.toList().filterNotNull())

    var yPred = naiveBayesClassifier.predictSimple(xValidVec)
    var result = yPred.zip(yTest.toList()) { a, b -> if (a == b) 1 else 0 }.average()
    println("Result of NaiveBayes: $result")

    // logistic regression
    val logisticRegression = LogisticRegression().asAutoOneHotClassifier<LogisticRegression, Int>()
    logisticRegression.fit(xTrainVec, yTrain.toList().filterNotNull())
    yPred = naiveBayesClassifier.predictSimple(xValidVec)
    result = yPred.zip(yTest.toList()) { a, b -> if (a == b) 1 else 0 }.average()
    println("Result of LogisticRegression: $result")
}