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
 review_final = review.select(review['business_id'].alias("bus_id_rev"), review['user_id'], review['user_id'], review['stars'], review['text'])
 rev = review.select(review['business_id'].alias("bus_id"), review['text'])
 bus_rev = bus.join(rev, rev['bus_id']==bus['business_id']).filter(bus['business_id']=='m2xeKBhS0szlm7xfU5b8ew')
 bus_count = bus_rev.select(functions.count(bus_rev['business_id'])).collect()
 
 
 food = spark.read.csv(input3, header = True, schema = food_schema)
 fd = food.select(food['business_id'].alias("bus_id_fd"), food['text'], food['food'].alias("food_rating"), (functions.regexp_extract(food['prob1'], '(.)(\d+.\d+)(.)', 2)).alias("f_prob1"), (functions.regexp_extract(food['prob2'], '(.)(\d+.\d+)(.)', 2)).alias("f_prob2"), (functions.regexp_extract(food['prob3'], '(.)(\d+.\d+)(.)', 2)).alias("f_prob3"))
 bus_fd = bus.join(fd, fd['bus_id_fd']==bus['business_id'])
 fd_business1 = bus_fd.select(bus_fd['bus_id_fd'], (bus_fd['food_rating']-bus_fd['food_rating']+1).alias("f_positive"), (bus_fd['food_rating']-bus_fd['food_rating']+0).alias("f_neutral"), (bus_fd['food_rating']-bus_fd['food_rating']+0).alias("f_negative")).filter(bus_fd['food_rating']==1)
 fd_business2 = bus_fd.select(bus_fd['bus_id_fd'], (bus_fd['food_rating']-bus_fd['food_rating']+0).alias("f_positive"), (bus_fd['food_rating']-bus_fd['food_rating']+1).alias("f_neutral"), (bus_fd['food_rating']-bus_fd['food_rating']+0).alias("f_negative")).filter(bus_fd['food_rating']==2)
 fd_business3 = bus_fd.select(bus_fd['bus_id_fd'], (bus_fd['food_rating']-bus_fd['food_rating']+0).alias("f_positive"), (bus_fd['food_rating']-bus_fd['food_rating']+0).alias("f_neutral"), (bus_fd['food_rating']-bus_fd['food_rating']+1).alias("f_negative")).filter(bus_fd['food_rating']==3)
 fd_bus = fd_business1.unionAll(fd_business2)
 fd_business = fd_bus.unionAll(fd_business3)
 fd_group = fd_business.groupby(fd_business['bus_id_fd']).agg(functions.sum(fd_business['f_positive']).alias("f_positive"), functions.sum(fd_business['f_neutral']).alias("f_neutral"), functions.sum(fd_business['f_negative']).alias("f_negative")) 
 fd_count = fd_group.select(functions.count(fd_group['bus_id_fd'])).collect()
 fd_c1 = fd_group.select(functions.sum(fd_group['f_positive'])).collect()
 fd_c2 = fd_group.select(functions.sum(fd_group['f_neutral'])).collect()
 fd_c3 = fd_group.select(functions.sum(fd_group['f_negative'])).collect()

 price = spark.read.csv(input5, header = True, schema = price_schema)
 pr = price.select(price['business_id'].alias("bus_id_pr"), price['text'], price['price'].alias("price_rating"), (functions.regexp_extract(price['p_prob1'], '(.)(\d+.\d+)(.)', 2)).alias("p_prob1"), (functions.regexp_extract(price['p_prob2'], '(.)(\d+.\d+)(.)', 2)).alias("p_prob2"), (functions.regexp_extract(price['p_prob3'], '(.)(\d+.\d+)(.)', 2)).alias("p_prob3"))
 bus_pr = bus.join(pr, pr['bus_id_pr']==bus['business_id'])
 pr_business1 = bus_pr.select(bus_pr['bus_id_pr'], (bus_pr['price_rating']-bus_pr['price_rating']+1).alias("p_positive"), (bus_pr['price_rating']-bus_pr['price_rating']+0).alias("p_neutral"), (bus_pr['price_rating']-bus_pr['price_rating']+0).alias("p_negative")).filter(bus_pr['price_rating']==1)
 pr_business2 = bus_pr.select(bus_pr['bus_id_pr'], (bus_pr['price_rating']-bus_pr['price_rating']+0).alias("p_positive"), (bus_pr['price_rating']-bus_pr['price_rating']+1).alias("p_neutral"), (bus_pr['price_rating']-bus_pr['price_rating']+0).alias("p_negative")).filter(bus_pr['price_rating']==2)
 pr_business3 = bus_pr.select(bus_pr['bus_id_pr'], (bus_pr['price_rating']-bus_pr['price_rating']+0).alias("p_positive"), (bus_pr['price_rating']-bus_pr['price_rating']+0).alias("p_neutral"), (bus_pr['price_rating']-bus_pr['price_rating']+1).alias("p_negative")).filter(bus_pr['price_rating']==3)
 pr_bus = pr_business1.unionAll(pr_business2)
 pr_business = pr_bus.unionAll(pr_business3)
 pr_group = pr_business.groupby(pr_business['bus_id_pr']).agg(functions.sum(pr_business['p_positive']).alias("p_positive"), functions.sum(pr_business['p_neutral']).alias("p_neutral"), functions.sum(pr_business['p_negative']).alias("p_negative")) 
 pr_count = pr_group.select(functions.count(pr_group['bus_id_pr'])).collect()
 pr_c1 = pr_group.select(functions.sum(pr_group['p_positive'])).collect()
 pr_c2 = pr_group.select(functions.sum(pr_group['p_neutral'])).collect()
 pr_c3 = pr_group.select(functions.sum(pr_group['p_negative'])).collect()
 
 service = spark.read.csv(input4, header = True, schema = service_schema)
 sr = service.select(service['business_id'].alias("bus_id_sr"), service['text'], service['service'].alias("service_rating"), (functions.regexp_extract(service['s_prob1'], '(.)(\d+.\d+)(.)', 2)).alias("s_prob1"), (functions.regexp_extract(service['s_prob2'], '(.)(\d+.\d+)(.)', 2)).alias("s_prob2"), (functions.regexp_extract(service['s_prob3'], '(.)(\d+.\d+)(.)', 2)).alias("s_prob3"))
 bus_sr = bus.join(sr, sr['bus_id_sr']==bus['business_id'])
 sr_business1 = bus_sr.select(bus_sr['bus_id_sr'], (bus_sr['service_rating']-bus_sr['service_rating']+1).alias("s_positive"), (bus_sr['service_rating']-bus_sr['service_rating']+0).alias("s_neutral"), (bus_sr['service_rating']-bus_sr['service_rating']+0).alias("s_negative")).filter(bus_sr['service_rating']==1)
 sr_business2 = bus_sr.select(bus_sr['bus_id_sr'], (bus_sr['service_rating']-bus_sr['service_rating']+0).alias("s_positive"), (bus_sr['service_rating']-bus_sr['service_rating']+1).alias("s_neutral"), (bus_sr['service_rating']-bus_sr['service_rating']+0).alias("s_negative")).filter(bus_sr['service_rating']==2)
 sr_business3 = bus_sr.select(bus_sr['bus_id_sr'], (bus_sr['service_rating']-bus_sr['service_rating']+0).alias("s_positive"), (bus_sr['service_rating']-bus_sr['service_rating']+0).alias("s_neutral"), (bus_sr['service_rating']-bus_sr['service_rating']+1).alias("s_negative")).filter(bus_sr['service_rating']==3)
 sr_bus = sr_business1.unionAll(sr_business2)
 sr_business = sr_bus.unionAll(sr_business3)
 sr_group = sr_business.groupby(sr_business['bus_id_sr']).agg(functions.sum(sr_business['s_positive']).alias("s_positive"), functions.sum(sr_business['s_neutral']).alias("s_neutral"), functions.sum(sr_business['s_negative']).alias("s_negative")) 
 sr_count = sr_group.select(functions.count(sr_group['bus_id_sr'])).collect()
 sr_c1 = sr_group.select(functions.sum(sr_group['s_positive'])).collect()
 sr_c2 = sr_group.select(functions.sum(sr_group['s_neutral'])).collect()
 sr_c3 = sr_group.select(functions.sum(sr_group['s_negative'])).collect()

 bus_fd_pr= bus.join(fd_group, fd_group['bus_id_fd']==bus['business_id'])
 bus_fd_final = bus_fd_pr.select(bus_fd_pr['business_id'],bus_fd_pr['name'],bus_fd_pr['latitude'],bus_fd_pr['longitude'], bus_fd_pr['categories'], bus_fd_pr['stars'], bus_fd_pr['review_count'],bus_fd_pr['f_positive'],bus_fd_pr['f_neutral'],bus_fd_pr['f_negative'])
 bus_pr_fd = bus_fd_final.join(pr_group, pr_group['bus_id_pr']==bus_fd_pr['business_id'])
 bus_fd_pr_final = bus_pr_fd.select(bus_pr_fd['business_id'],bus_pr_fd['name'],bus_pr_fd['latitude'],bus_pr_fd['longitude'], bus_pr_fd['categories'], bus_pr_fd['stars'], bus_pr_fd['review_count'],bus_pr_fd['f_positive'],bus_pr_fd['f_neutral'],bus_pr_fd['f_negative'],bus_pr_fd['p_positive'],bus_pr_fd['p_neutral'],bus_pr_fd['p_negative'])
 bus_pr_fd_sr = bus_fd_pr_final.join(sr_group, sr_group['bus_id_sr']==bus_fd_pr_final['business_id'])
 bus_fd_pr_sr_final = bus_pr_fd_sr.select(bus_pr_fd_sr['business_id'],bus_pr_fd_sr['name'],bus_pr_fd_sr['latitude'],bus_pr_fd_sr['longitude'], bus_pr_fd_sr['categories'], bus_pr_fd_sr['stars'], bus_pr_fd_sr['review_count'],bus_pr_fd_sr['f_positive'],bus_pr_fd_sr['f_neutral'],bus_pr_fd_sr['f_negative'],bus_pr_fd_sr['p_positive'],bus_pr_fd_sr['p_neutral'],bus_pr_fd_sr['p_negative'],bus_pr_fd_sr['s_positive'],bus_pr_fd_sr['s_neutral'],bus_pr_fd_sr['s_negative'])
 
 vader = spark.read.json(input6)
 vd = vader.select(vader['id'], (vader['composite']).alias("v_composite"), (vader['positive']).alias("v_positive"), (vader['neutral']).alias("v_neutral"), (vader['negative']).alias("v_negative"))
 vd_group = vd.groupby(vader['id']).agg(functions.round(functions.sum(vd['v_composite'])/functions.count(vd['id']),2).alias("v_composite"), functions.round(functions.sum(vd['v_positive'])/functions.count(vd['id']),2).alias("v_positive"),functions.round(functions.sum(vd['v_neutral'])/functions.count(vd['id']),2).alias("v_neutral"), functions.round(functions.sum(vd['v_negative'])/functions.count(vd['id']),2).alias("v_negative")) 
 bus_pr_fd_sr_vd = bus_fd_pr_sr_final.join(vd_group, vd_group['id']==bus_fd_pr_sr_final['business_id'])
 bus_fd_pr_sr_vd_final = bus_pr_fd_sr_vd.select(bus_pr_fd_sr_vd['business_id'],bus_pr_fd_sr_vd['name'],bus_pr_fd_sr_vd['latitude'],bus_pr_fd_sr_vd['longitude'], bus_pr_fd_sr_vd['categories'], bus_pr_fd_sr_vd['stars'], bus_pr_fd_sr_vd['review_count'],bus_pr_fd_sr_vd['f_positive'],bus_pr_fd_sr_vd['f_neutral'],bus_pr_fd_sr_vd['f_negative'],bus_pr_fd_sr_vd['p_positive'],bus_pr_fd_sr_vd['p_neutral'],bus_pr_fd_sr_vd['p_negative'],bus_pr_fd_sr_vd['s_positive'],bus_pr_fd_sr_vd['s_neutral'],bus_pr_fd_sr_vd['s_negative'],bus_pr_fd_sr_vd['v_composite'],bus_pr_fd_sr_vd['v_positive'],bus_pr_fd_sr_vd['v_neutral'],bus_pr_fd_sr_vd['v_negative'])
 vd_c3 = bus_fd_pr_sr_vd_final.select(functions.count(bus_fd_pr_sr_final['business_id'])).collect()

 print(bus_fd_pr_sr_vd_final.show(10))
 #bus_fd_pr_sr_vd_final.write.format("org.apache.spark.sql.cassandra").options(table='yelp_business_combined', keyspace='amahadev').save()
 #bus_fd_pr_sr_vd_final.write.csv(output+"final_CSV")
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