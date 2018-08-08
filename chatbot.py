from pyspark.sql import SQLContext
from pyspark.sql.types import StructField,StructType,StringType,FloatType
from pyspark import SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover,CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

sc = SparkContext()
sqlContext = SQLContext(sc)
language = "english"
data = None
regexTokenizer = None
stopwordsRemover = None
countVectors = None
dataset = None
lrModel = None
pipeline = None
pipelineFit = None

def init():

    global data,regexTokenizer,stopwordsRemover,countVectors
    
    # Read the seed training data into the system
    
    data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('seeddata.csv')
    
    # Contains a list of columns we don't care about
    
    drop_list = []
    
    # Grab data in all columns that we care about
    data = data.select([column for column in data.columns if column not in drop_list])

    data.show(5)
    data.printSchema()

    # regular expression tokenizer
    regexTokenizer = RegexTokenizer(inputCol="question",outputCol="words", pattern="\\W")

    # stop words
    add_stopwords = None

    # Load some default stopwords
    if add_stopwords == None:
        add_stopwords = StopWordsRemover.loadDefaultStopWords(language)

        # Remove certain stop words that provide context for our use case
        needed_stopwords = ['what','when','where','why']
        for x in needed_stopwords:
            add_stopwords.remove(x)
        print("stop words:\n {}".format(add_stopwords))
    
    stopwordsRemover = StopWordsRemover(inputCol="words",outputCol="filtered").setStopWords(add_stopwords)

    # bag of words count
    countVectors = CountVectorizer(inputCol="filtered",outputCol="features", vocabSize=15, minDF=1)


def trainFeatureExtraction():

    global dataset, pipeline, pipelineFit

    # Encode a string column of labels ordered by label frequencies

    label_stringIdx = StringIndexer(inputCol = "category", outputCol="label").setHandleInvalid("keep") 

    pipeline = Pipeline(stages=[regexTokenizer,stopwordsRemover,countVectors, label_stringIdx])

    # Fit the pipeline to training data
    pipelineFit = pipeline.fit(data)
    dataset = pipelineFit.transform(data)
    dataset.show(10)

def scoringFeatureExtraction(input):

    
    data = [(input,'','')]
    #Make Dataframe from data
    df  = sqlContext.createDataFrame(data,['question','category','answer'])
    #df.collect()
    #df.show()
    #df.printSchema()

    #df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('score.csv')

    # Predict on the sample data
    dataset = pipelineFit.transform(df)

    return dataset


def train(interations):

    global testData, lrModel

    # Split the data into 70% training data
    #(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
    
    # Train with all the data because we have very little data
    trainingData  = dataset
    
    lr = LogisticRegression(maxIter=interations, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)

# Predict using the testData if traintest is False.  Otherwise, predict based on the data parameter, which should be a dataframe
def predict(traintest=False,data=None,question=None):

    if traintest == True:
        inputData = testData
    elif data != None:
        print("Prediction using datafrome:\n")
        inputData = data
        predictions = lrModel.transform(inputData)
        predictions.show()
    else:
        print("Prediction using a string:\n")
        inputData = scoringFeatureExtraction(question)
        inputData.show()
        prediction = lrModel.transform(inputData)
        return prediction.head().prediction

def driver():
    while True:
        try:

            myQuestion = input(":=> ")
            answer=predict(question=myQuestion)
            print(":=> {}".format(answer))
        except ValueError:
            print(":=> Sorry, I didn't understand that.")
            continue

if __name__ == "__main__":
    print("Afrotech Chatbot\n")
    init()
    trainFeatureExtraction()
    train(100)
    clientData = dataset.sample(False,0.30)
    #predict(data=clientData)
    #myQuestion = "what is afrotech"
    #predict(question=myQuestion)
    #for x in range(100,1000,20):
    #    train(x)
    #    predict(data=clientData)
    driver()
