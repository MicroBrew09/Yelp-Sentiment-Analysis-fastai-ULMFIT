import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd
from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from pyspark.sql.functions import regexp_extract
from langdetect import detect_langs

def sentiment_analyzer_scores(sentence):
    a = sentence[1]
    score = analyzer.polarity_scores(a)
    return(sentence[0], score['neg'],score['neu'],score['pos'],score['compound']) #, sentence[2]) 
    
def main(inputs, output):
    observation_schema = types.StructType([
       types.StructField('id', types.StringType()),
       types.StructField('text', types.StringType()),
       types.StructField('score', types.StringType()),
])

    df = spark.read.json(inputs)
    df_text = df.select('business_id','text')
    rddstuff = df_text.rdd
    finalrddstuff = rddstuff.map(sentiment_analyzer_scores)
    RDD_t = spark.createDataFrame(finalrddstuff)
    RDD_t = RDD_t.withColumnRenamed('_1', 'id')
    RDD_t = RDD_t.withColumnRenamed('_2', 'negative')
    RDD_t = RDD_t.withColumnRenamed('_3', 'neutral')
    RDD_t = RDD_t.withColumnRenamed('_4', 'positive')
    RDD_t = RDD_t.withColumnRenamed('_5', 'composite')
    RDD_t.write.json(output)
    RDD_t.write.csv(output+str(1))


if __name__ == '__main__':
    inputs = sys.argv[1] #Either service or price
    output = sys.argv[2]
    main(inputs, output)
