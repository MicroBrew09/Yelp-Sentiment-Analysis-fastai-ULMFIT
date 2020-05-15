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
cluster_seeds = ['199.60.17.32', '199.60.17.65']
spark = SparkSession.builder.appName('Spark Cassandra example') \
    .config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
# add more functions as necessary

food_schema = types.StructType([
    types.StructField('_c0', types.StringType()),
    types.StructField('business_id', types.StringType()),
    types.StructField('text', types.StringType()),
    types.StructField('food', types.IntegerType()),
    types.StructField('prob1', types.StringType()),
    types.StructField('prob2', types.StringType()),
    types.StructField('prob3', types.StringType()),
])
price_schema = types.StructType([
    types.StructField('_c0', types.StringType()),
    types.StructField('business_id', types.StringType()),
    types.StructField('text', types.StringType()),
    types.StructField('price', types.IntegerType()),
    types.StructField('p_prob1', types.StringType()),
    types.StructField('p_prob2', types.StringType()),
    types.StructField('p_prob3', types.StringType()),
])
service_schema = types.StructType([
    types.StructField('_c0', types.StringType()),
    types.StructField('business_id', types.StringType()),
    types.StructField('text', types.StringType()),
    types.StructField('service', types.IntegerType()),
    types.StructField('s_prob1', types.StringType()),
    types.StructField('s_prob2', types.StringType()),
    types.StructField('s_prob3', types.StringType()),
])

vader_schema = types.StructType([
    types.StructField('composite', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('negative', types.LongType()),
    types.StructField('neutral', types.LongType()),
    types.StructField('positive', types.LongType()),
])

def main(input1,input2,input3,input4,input5,input6,output):
    # main logic starts here
 business = spark.read.json(input1)
 bus = business.select(business['business_id'],business['name'],business['latitude'],business['longitude'], business['categories'], business['stars'], business['review_count']).filter(business['categories'].contains("Restaurants,")).filter(business['city'].contains("Toronto"))
 review = spark.read.json(input2)
 review_final = review.select(review['business_id'].alias("bus_id_rev"), review['user_id'], review['user_id'], review['stars'].alias("review_stars"), review['text'])
 bus_rev = bus.join(review_final, review_final['bus_id_rev']==bus['business_id'])
 rev_txt = bus_rev.select(bus_rev['business_id'], bus_rev['name'],bus_rev[latitude],bus_rev[longitude])
 
 food = spark.read.csv(input3, header = True, schema = food_schema)
 fd = food.select(food['business_id'].alias("bus_id_fd"), food['text'].alias("text"), functions.length(food['text']).alias("f_rev"), food['food'].alias("food_rating"), (functions.regexp_extract(food['prob1'], '(.)(\d+.\d+)(.)', 2)).alias("f_prob_pos"), (functions.regexp_extract(food['prob2'], '(.)(\d+.\d+)(.)', 2)).alias("f_prob_neu"), (functions.regexp_extract(food['prob3'], '(.)(\d+.\d+)(.)', 2)).alias("f_prob_neg"))
 rev_fd = rev_txt.join(fd, fd['bus_id_fd']==rev_txt['business_id'])
 rev_fd_f = rev_fd.select(rev_fd[latitude],rev_fd[longitude],rev_fd['business_id'], rev_fd['name'], rev_fd['text'], rev_fd['f_rev'], rev_fd['food_rating'], rev_fd['f_prob_pos'], rev_fd['f_prob_neu'], rev_fd['f_prob_neg']).distinct()

 price = spark.read.csv(input5, header = True, schema = price_schema)
 pr = price.select(price['business_id'].alias("bus_id_pr"), price['text'].alias("price_text"), functions.length(price['text']).alias("p_rev"), price['price'].alias("price_rating"), (functions.regexp_extract(price['p_prob1'], '(.)(\d+.\d+)(.)', 2)).alias("p_prob_pos"), (functions.regexp_extract(price['p_prob2'], '(.)(\d+.\d+)(.)', 2)).alias("p_prob_neu"), (functions.regexp_extract(price['p_prob3'], '(.)(\d+.\d+)(.)', 2)).alias("p_prob_neg"))
 rev_pr = rev_fd_f.join(pr, pr['bus_id_pr']==rev_fd_f['business_id'])
 rev_fd_pr = rev_pr.select(rev_pr[longitude],rev_pr[latitude],rev_pr['business_id'], rev_pr['name'], rev_pr['p_rev'], rev_pr['text'], rev_pr['food_rating'], rev_pr['f_prob_pos'], rev_pr['f_prob_neu'], rev_pr['f_prob_neg'], rev_pr['price_rating'], rev_pr['p_prob_pos'], rev_pr['p_prob_neu'], rev_pr['p_prob_neg']).filter(rev_pr['p_rev'] == rev_pr['f_rev']).distinct()
 

 service = spark.read.csv(input4, header = True, schema = service_schema)
 sr = service.select(service['business_id'].alias("bus_id_sr"), service['text'].alias("service_text"), functions.length(service['text']).alias("s_rev"), service['service'].alias("service_rating"), (functions.regexp_extract(service['s_prob1'], '(.)(\d+.\d+)(.)', 2)).alias("s_prob_pos"), (functions.regexp_extract(service['s_prob2'], '(.)(\d+.\d+)(.)', 2)).alias("s_prob_neu"), (functions.regexp_extract(service['s_prob3'], '(.)(\d+.\d+)(.)', 2)).alias("s_prob_neg"))
 rev_sr = rev_fd_pr.join(sr, sr['bus_id_sr']==rev_fd_pr['business_id'])
 print(rev_fd_pr_sr.show(10))
 rev_fd_pr_sr = rev_sr.select(rev_sr[longitude],rev_sr[latitude],rev_sr['business_id'], rev_sr['name'], rev_sr['text'], rev_sr['food_rating'], rev_sr['f_prob_pos'], rev_sr['f_prob_neu'], rev_sr['f_prob_neg'], rev_sr['price_rating'], rev_sr['p_prob_pos'], rev_sr['p_prob_neu'], rev_sr['p_prob_neg'], rev_sr['service_rating'], rev_sr['s_prob_pos'], rev_sr['s_prob_neu'], rev_sr['s_prob_neg']).filter(rev_sr['s_rev'] == rev_sr['p_rev']).filter(rev_sr['s_prob_pos']!=rev_sr['s_prob_neu']).filter(rev_sr['s_prob_pos']!=rev_sr['s_prob_neg']).distinct()
 
 print(rev_fd_pr_sr.show(10))
 #rev_fd_pr_sr.write.format("org.apache.spark.sql.cassandra").options(table='review_sample', keyspace='amahadev').save()
 rev_fd_pr_sr.write.csv(output+"rev_list_CSV")
 #bus_fd_pr_sr_vd_final.write.json(output+"final_JSON")

if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    input3 = sys.argv[3]
    input4 = sys.argv[4]
    input5 = sys.argv[5]
    input6 = sys.argv[6]        
    output = sys.argv[7]
    main(input1,input2,input3,input4,input5,input6,output)