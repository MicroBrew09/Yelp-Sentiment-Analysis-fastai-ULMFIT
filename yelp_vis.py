# Map icons adapted from code from https://docs.mapbox.com/mapbox-gl-js/example/custom-marker-icons/

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext 
from pyspark.sql import types


data_schema = types.StructType([
    types.StructField('business_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('categories', types.StringType()),
    types.StructField('stars', types.FloatType()),
    types.StructField('review_count', types.LongType()),
    types.StructField('f_positive', types.LongType()),
    types.StructField('f_neutral', types.LongType()),
    types.StructField('f_negative', types.LongType()),
    types.StructField('p_positive', types.LongType()),
    types.StructField('p_neutral', types.LongType()),
    types.StructField('p_negative', types.LongType()),
    types.StructField('s_positive', types.LongType()),
    types.StructField('s_neutral', types.LongType()),
    types.StructField('s_negative', types.LongType()),
    types.StructField('v_composite', types.FloatType()),
    types.StructField('v_positive', types.FloatType()),
    types.StructField('v_neutral', types.FloatType()),
    types.StructField('v_negative', types.FloatType()),

])


def main(inputs):
	rest_data = spark.read.json("Output_Files/final.json", schema=data_schema)
	rest_data.createTempView("rest_data")
	rest_main = spark.sql("SELECT * FROM rest_data WHERE rest_data.name = '" + inputs + "'") #This is our main restaurant
	rest_list = rest_main.collect()
	if len(rest_list) > 0:
		name = rest_list[0][1]
		latitude = rest_list[0][2]
		longitude = rest_list[0][3]
		categories = rest_list[0][4]
		stars = rest_list[0][5]
		review_count = rest_list[0][6]
		f_pos = rest_list[0][7]
		f_neu = rest_list[0][8]
		f_neg = rest_list[0][9]
		p_pos = rest_list[0][10]
		p_neu = rest_list[0][11]
		p_neg = rest_list[0][12]
		s_pos = rest_list[0][13]
		s_neu = rest_list[0][14]
		s_neg = rest_list[0][15]
		v_comp = rest_list[0][16]
		sql_statement = "SELECT name, latitude, longitude FROM rest_data WHERE (rest_data.name != '" + name + "' ) AND (" #Get the competing restaurants within the area
		for i in categories.split(', '):
			if i != "Food" and i != "Restaurants":
				sql_statement += "rest_data.categories RLIKE '^" + i + "' OR " #Add the categories to search for
		if sql_statement.endswith("OR "):
			sql_statement = sql_statement[:-4]
		sql_statement += ")"
		rest_comp = spark.sql(sql_statement)
		comp_list = rest_comp.collect()
		comp_list_close = []
		for i in comp_list:
			if (i[1] >= (latitude - 0.05)) and (i[1] <= (latitude + 0.05)) and (i[2] >= (longitude - 0.05)) and (i[2] <= (longitude + 0.05)):
				comp_list_close.append(i)
		print(sql_statement)
		html_scr = "<!DOCTYPE html><html lang='en'><head> <title>Restaurant Analytics</title> <meta charset='utf-8'> <meta name='viewport' content='width=device-width, initial-scale=1'> <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css'> <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js'></script> <script src='https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js'></script> <script src='https://api.tiles.mapbox.com/mapbox-gl-js/v1.5.0/mapbox-gl.js'></script> <link href='https://api.tiles.mapbox.com/mapbox-gl-js/v1.5.0/mapbox-gl.css' rel='stylesheet'/> <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'></head><style>#map{position: absolute;top: 87%;bottom: 0;width: 60%;height: 80%;margin-left: 200px}</style><body style='background-color: black; color: white; font-family: garamond'><div class=container-fluid><div class='jumbotron jumbotron-fluid' style='background-image: url(https://cdn1.analytics.hbs.edu/content/0e06dff0f9384e3fba3e332ac40100df/BizAnalytics_vs_Intelligence-Hero.jpg);background-size:cover'><div class='container'><h1 class='display-4'>Restaurant Analytics</h1><h1>" + name + "</h1></div></div></div><div><a href='https://public.tableau.com/views/yelpanalysis/FinalData?:display_count=y&publish=yes&:origin=viz_share_link'>Click Here for Tableau 1</a></div><hr><div class=container style='text-align:center'><div class=row> <div class=col-sm-8><div class=row><h4>We analyzed your text based reviews.<br>Numbers indicate the count for each category.</h4></div><div class=col-sm-4> <h2>Price</h2> </div><div class=col-sm-4><h2>Service</h2> </div><div class=col-sm-4><h2>Food</h2></div></div><div class=col-sm-4><h4>Overall sentiment score runs from<br><b>-1 </b>(very negative) to <b>1</b> (very positive)</h4><h2>Overall Sentiment:</h2> </div></div><div class=row> <div class=col-sm-8> <div class=row> <div class='col-sm-4'> <div class=row> <div class=col-sm-4> <i class='fa fa-smile-o' style='font-size:35px; color:#bfff7e'></i></div><div class=col-sm-4><i class='fa fa-meh-o' style='font-size:35px; color:#e9a2ff'></i></div><div class=col-sm-4><i class='fa fa-frown-o' style='font-size:35px; color:#ff8f6d'></i></div></div><div class=row> <div class=col-sm-4> <h3>" + str(p_pos) + "</h3> </div><div class=col-sm-4> <h3>" + str(p_neu) + "</h3> </div><div class=col-sm-4> <h3>" + str(p_neg) + "</h3> </div></div></div><div class=col-sm-4>    <div class=row><div class=col-sm-4><i class='fa fa-smile-o' style='font-size:35px; color:#bfff7e'></i></div><div class=col-sm-4><i class='fa fa-meh-o' style='font-size:35px; color:#e9a2ff'></i></div><div class=col-sm-4><i class='fa fa-frown-o' style='font-size:35px; color:#ff8f6d'></i></div></div><div class=row> <div class=col-sm-4> <h3>" + str(s_pos) + "</h3> </div><div class=col-sm-4> <h3>" + str(s_neu) + "</h3> </div><div class=col-sm-4> <h3>" + str(s_neg) + "</h3> </div></div></div><div class='col-sm-4'> <div class=row> <div class=col-sm-4><i class='fa fa-smile-o' style='font-size:35px; color:#bfff7e'></i></div><div class=col-sm-4><i class='fa fa-meh-o' style='font-size:35px; color:#e9a2ff'></i></div><div class=col-sm-4><i class='fa fa-frown-o' style='font-size:35px; color:#ff8f6d'></i></div></div><div class=row> <div class=col-sm-4> <h3>" + str(f_pos) + "</h3> </div><div class=col-sm-4> <h3>" + str(f_neu) + "</h3> </div><div class=col-sm-4> <h3>" + str(f_neg) + "</h3> </div></div></div></div></div><div class=col-sm-4> <h1><b>" + str("{0:.2f}".format(v_comp)) + "</b></h1> </div></div><div class=row><div class=col-sm-4> <h2>Number of Reviews Analyzed:</h2> </div><div class=col-sm-4> <h2>Total Number of Ratings:</h2> </div><div class='col-sm-4'> <h2>Average <br><i class='fa fa-star' style='font-size:35px'></i>:</h2> </div></div><div class=row><div class=col-sm-4> <h1><b>" + str(p_pos + p_neu + p_neg) + "</b></h1> </div><div class=col-sm-4><h1><b>" + str(review_count) + "</b></h1></div><div class='col-sm-4'> <h1><b>" + str(stars) + "</b></h1> </div></div></div><hr><div class=container> <h1>Similar restaurants in your area:</h1> </div><div class=container-fluid><div id=map style='margin-right: 400px'></div> <script>mapboxgl.accessToken='pk.eyJ1IjoiY3JpZGRlbCIsImEiOiJjazNnNnlvZnMwY290M25vMmh4MjE2eDU3In0.UBwZQRbedsW1Ezj4IV-fvQ'; var map=new mapboxgl.Map({container: 'map', style: 'mapbox://styles/mapbox/streets-v11', center: [" + str(longitude) + "," + str(latitude) + "], zoom: 13}); var geojson={'type': 'FeatureCollection', 'features': [ "

		for i in comp_list:
			new_name = i[0].replace("'", "")

			html_scr += "{'type': 'Feature','properties': {'message':'" + new_name + "'},'geometry': {'type': 'Point', 'coordinates': [" + str(i[2]) + "," + str(i[1]) + "]} }, "	
		
		html_scr += "]}; map.on('load', function(){map.loadImage('https://i.imgur.com/MK4NUzI.png', function(error, image){if (error) throw error; map.addImage('marker', image); map.addLayer({'id': 'points', 'type': 'symbol', 'source':{'type': 'geojson', 'data':{'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'message':'" + name + "'}, 'geometry':{'type': 'Point', 'coordinates': [" + str(longitude) + "," + str(latitude) + "]}}]}}, 'layout':{'icon-image': 'marker', 'icon-size': 1.5}});}); geojson.features.forEach(function(marker){var el=document.createElement('div'); el.className='marker_2'; el.style.backgroundImage='url(https://www.iconsdb.com/icons/preview/deep-pink/map-marker-2-xxl.png)';el.style.width = '40px';el.style.height = '40px'; el.style.backgroundSize='contain';el.addEventListener('click', function(){window.alert(marker.properties.message);}); new mapboxgl.Marker(el) .setLngLat(marker.geometry.coordinates) .addTo(map);});}); </script> </div></div></body></html>"

		f = open("yelp_vis_temp.html", "w")
		f.write(html_scr)
		f.close()

	else:	
		print("No data available. Try another restaurant.")

if __name__ == '__main__':
	inputs = sys.argv[1]
	main(inputs)

