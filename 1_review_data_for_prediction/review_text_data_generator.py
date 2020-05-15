import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import pandas as pd
import numpy as np
import string
from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
# add more functions as necessary

def language_filter(X):
 if (detect(X[1]) == 'en'):
  return(X) 	

def main(input1,input2,output):
    # main logic starts here
 business = spark.read.json(input1)
 bus = business.select(business['business_id'],).filter(business['categories'].contains("Restaurants,")).filter(business['city'].contains("Toronto"))
 review = spark.read.json(input2)
 rev = review.select(review['business_id'].alias("bus_id"), review['text'])
 bus_rev = bus.join(rev, rev['bus_id']==bus['business_id'])
 bus_text = bus_rev.select(bus_rev['business_id'], (functions.lower(functions.regexp_replace(bus_rev['text'],"[-()\"#/@;:<>{}`+=~|.!?,]\n'[^A-Z0-9-]", ''))).alias("text"))
 bus_count = bus_text.select(functions.count(bus_text['business_id'])).collect()
 print(bus_count)
 bus_text.write.csv(output)
 bus_text.write.json(output+str(2))
 review_rdd = bus_text.rdd.map(tuple)
 review_rdd.saveAsTextFile(output+str(1))
 print(bus_text.show(20))
 print(review_rdd.take(20))

if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    output = sys.argv[3]
    main(input1,input2,output)

