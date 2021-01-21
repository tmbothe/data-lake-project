import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date, LongType as long

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    This function reads song's data from s3, load in dataframe
    then, extract songs, artists table and write them back to s3 
    in parquet format
    """
    
    #Defining a schema for song's data
    
    song_schema = R([
    Fld("artist_id" ,Str()),
    Fld("artist_latitude",Dbl()),
    Fld("artist_location",Str()),
    Fld("artist_longitude",Dbl()),
    Fld("artist_name",Str()),
    Fld("duration",Dbl()),
    Fld("num_songs",long()),
    Fld("song_id",Str()),
    Fld("title",Str()),
    Fld("year",long())
    ])

    #input_data = 's3a://udacity-dend'
    songs_path = 'song_data/*/*/*/*.json'
    song_data  =  os.path.join(input_data,songs_path)

    # read song data file
    df = spark.read.json(song_data,schema=song_schema)

    # extract columns to create songs table
    songs_table = df.select('song_id','title','artist_id','year','duration').drop_duplicates()

    # write songs table to parquet files partitioned by year and artist
    destination_song = os.path.join(output_data,"songs.parquet")
    songs_table.write.mode('overwrite').partitionBy("year","artist_id").parquet(destination_song)

    # extract columns to create artists table
    artists_table = df.selectExpr('artist_id','artist_name as name','artist_location as location',
                                  'artist_latitude as latitude', 'artist_longitude as longitude').drop_duplicates()

    # write artists table to parquet files

    destination_artists = os.path.join(output_data,'artists.parquet')

    artists_table.write.mode('overwrite').parquet(destination_artists)


def process_log_data(spark, input_data, output_data):
    
    log_data = input_data + "log_data/*/*"
    df = spark.read.json(log_data)
    # filter by actions for song plays
    df = df.filter(df.page=='NextSong')

    # extract columns for users table    
    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level').drop_duplicates()

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(os.path.join(output_data,'users.parquet'))

    # create timestamp column from original timestamp column
    get_timestamp = F.udf(lambda ts:  str(int(int(ts)/1000))) 

    # create datetime column from original timestamp column
    df = df.withColumn("timestamp", get_timestamp(df.ts))

    # extract columns to create time table
    df = df.withColumn("start_time", F.to_timestamp(F.from_unixtime((col("ts") / 1000) , 'yyyy-MM-dd HH:mm:ss.SSS')).cast("Timestamp"))

    # write time table to parquet files partitioned by year and month
    time_table = df.select('start_time').withColumn('hour',hour('start_time'))
    time_table = time_table.withColumn('day',dayofmonth('start_time'))
    time_table = time_table.withColumn('week',weekofyear('start_time'))
    time_table = time_table.withColumn('month',month('start_time'))
    time_table = time_table.withColumn('year',year('start_time'))
    time_table = time_table.withColumn('weekday',date_format('start_time',"EEEE"))

    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').parquet(os.path.join(output_data,'time.parquet'))

    # read in song data to use for songplays table
    song_df = spark.read.parquet(os.path.join(output_data,"songs.parquet"))

    # extract columns from joined song and log datasets to create songplays table 
    temp = df.join(song_df,(df.song==song_df.title) ,how='inner' ).selectExpr('start_time','userId as user_id','artist_id','level','song_id','sessionId as session_id','location','userAgent as user_agent')
    temp= temp.withColumn('songplay_id',F.monotonically_increasing_id())
    songplays_table = temp.select('songplay_id', 'start_time', 'user_id', 'level', 'song_id', 'artist_id', 'session_id', 'location', 'user_agent')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode('overwrite').parquet(os.path.join(output_data,"songplays.parquet"))


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = ""
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
