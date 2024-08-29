import json
import sys
from math import sqrt, log
from operator import add
from pyspark import SparkContext, SparkConf
from xgboost import XGBRegressor
import time 
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def parse_photos(line):
    data = json.loads(line)
    return (data['business_id'], 1)  # 返回商家ID和照片计数1

def parse_checkins(line):
    data = json.loads(line)
    total_checkins = sum(data['time'].values())  # 累加所有时间段的check-in数
    return (data['business_id'], total_checkins)

def parse_business_rating(line):
    fields = line.split(',')
    return (fields[0], fields[1], float(fields[2]))  # For training data with stars

def parse_user_business(line):
    fields = line.split(',')
    return (fields[0], fields[1])  # For validation data without stars
    
def calculate_similarity_score(business_a, business_b, business_user_ratings):
    shared_users = business_users_map[business_a] & business_users_map[business_b]
    num_shared_users = len(shared_users)
    
    if num_shared_users <= 1:
        return 0.2 * (1.0 - abs(business_avg_ratings[business_a] - business_avg_ratings[business_b]) / 4.0)
    elif num_shared_users == 2:
        ratings_a = [float(business_user_ratings[business_a][u]) for u in shared_users]
        ratings_b = [float(business_user_ratings[business_b][u]) for u in shared_users]
        diff_squared_1 = (ratings_a[0] - ratings_b[0]) ** 2
        diff_squared_2 = (ratings_a[1] - ratings_b[1]) ** 2
        return 0.3 * (2.0 - 0.25 * (diff_squared_1 + diff_squared_2))
    else:
        ratings_a = [float(business_user_ratings[business_a][u]) for u in shared_users]
        ratings_b = [float(business_user_ratings[business_b][u]) for u in shared_users]
        mean_a, mean_b = sum(ratings_a) / num_shared_users, sum(ratings_b) / num_shared_users
        numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(ratings_a, ratings_b))
        denominator = sqrt(sum((x - mean_a) ** 2 for x in ratings_a)) * sqrt(sum((y - mean_b) ** 2 for y in ratings_b))
        pearson_correlation = numerator / denominator if denominator else 0.0
        return 0.5 * (pearson_correlation + 1.0)

def estimate_rating_item_based(user, business, user_to_business_map, business_to_user_ratings, similarity_scores):
    if user not in user_to_business_map:
        return 3.0
    if business not in business_users_map:
        return user_avg_ratings.get(user, 3.0)  
    
    similarities = []
    for business_rated in user_to_business_map[user]:
        if business_rated == business:
            continue
        pair_key = tuple(sorted([business, business_rated]))
        if pair_key not in similarity_scores:
            similarity_scores[pair_key] = calculate_similarity_score(business, business_rated, business_to_user_ratings)
        similarities.append((similarity_scores[pair_key], float(business_to_user_ratings[business_rated][user])))

    top_similarities = sorted(similarities, reverse=True)[:10]
    weighted_sum, total_weight = sum(score * rating for score, rating in top_similarities), sum(abs(score) for score, _ in top_similarities)
    return weighted_sum / total_weight if total_weight else user_avg_ratings.get(user, 3.0)

def parse_business(line):
    business = json.loads(line)
    return (business['business_id'], (business['stars'], business['review_count']))

def parse_review(line):
    review = json.loads(line)
    return (review['business_id'], (review['useful'], review['funny'], review['cool']))

def parse_user(line):
    user = json.loads(line)
    user_fans_squared = user['fans'] ** 2
    user_elite = 1 if user['elite'] != 'None' else 0

    # Calculate the sum of squares of all compliments
    compliments = [
        user['compliment_hot'],
        user['compliment_more'],
        user['compliment_profile'],
        user['compliment_cute'],
        user['compliment_list'],
        user['compliment_note'],
        user['compliment_plain'],
        user['compliment_cool'],
        user['compliment_funny'],
        user['compliment_writer'],
        user['compliment_photos']
    ]
    compliment_sum_squared = sum(x ** 2 for x in compliments)

    # Sum of squares of useful, cool, and funny
    ucf_sum_squared = user['useful'] ** 2 + user['cool'] ** 2 + user['funny'] ** 2

    return (user['user_id'], (
        user['review_count'],
        user['average_stars'],
        user_fans_squared,
        user_elite,
        compliment_sum_squared,
        ucf_sum_squared  # Keep only this composite feature
    ))

def get_user_features_partition(iterator):
    for user_info in iterator:
        user_id, (review_count, average_stars, fans_squared, elite_status, compliments_squared_sum, ucf_sum_squared) = user_info
        yield (user_id, [
            review_count, 
            average_stars, 
            fans_squared, 
            elite_status, 
            compliments_squared_sum,
            ucf_sum_squared  # Ensure this is passed through
        ])

def get_business_features_partition(iterator):
    for business_info in iterator:
        business_id, (business_data, grouped_reviews) = business_info
        stars, review_count = business_data
        useful, funny, cool = 0, 0, 0
        review_cnt = 0
        
        for review in grouped_reviews or []:
            useful += review[0]
            funny += review[1]
            cool += review[2]
            review_cnt += 1
        
        useful_avg = 0 if review_cnt == 0 else useful / review_cnt
        funny_avg = 0 if review_cnt == 0 else funny / review_cnt
        cool_avg = 0 if review_cnt == 0 else cool / review_cnt
        
        total_checkins = checkin_features.get(business_id, 0)
        total_photos = photo_features.get(business_id, 0)
        
        # 计算评论数、照片数和check-in数的平均值
        average_popularity = (review_count + total_photos + total_checkins) / 3
        
        yield (business_id, [stars, review_count, useful_avg, funny_avg, cool_avg, total_checkins, total_photos, average_popularity])




def build_features(line, business_features_broadcast, user_features_broadcast):
    parts = line.split(',')
    user_id, business_id = parts[0], parts[1]
    business_feat = business_features_broadcast.value.get(business_id, [0] * 11)
    user_feat = user_features_broadcast.value.get(user_id, [0] * 5)
    features = business_feat + user_feat
    
    if len(parts) > 2:
        rating = float(parts[2])
        return (user_id, business_id, features, rating)
    else:
        return (user_id, business_id, features)


if __name__ == "__main__":
    start_time = time.time() 
    
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    alpha = 0.1

    spark_conf = SparkConf().setAppName("HybridRecommendationSystem")
    sc = SparkContext(conf=spark_conf)

    training_data_path = folder_path + "yelp_train.csv"
    training_ratings_rdd = sc.textFile(training_data_path).filter(lambda x: not x.startswith('user_id')).map(parse_business_rating)
    validation_ratings_rdd = sc.textFile(test_file).filter(lambda x: not x.startswith('user_id')).map(parse_user_business)

    business_users_map = training_ratings_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    user_to_business_map = training_ratings_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    business_to_user_ratings = training_ratings_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collectAsMap()
    user_avg_ratings = training_ratings_rdd.map(lambda x: (x[0], x[2])).groupByKey().mapValues(lambda vals: sum(vals) / len(vals)).collectAsMap()
    business_avg_ratings = training_ratings_rdd.map(lambda x: (x[1], x[2])).groupByKey().mapValues(lambda vals: sum(vals) / len(vals)).collectAsMap()

    similarity_scores_broadcast = sc.broadcast({})
    
    business_data = sc.textFile(folder_path + "business.json").map(parse_business).cache()
    review_data = sc.textFile(folder_path + "review_train.json").map(parse_review).cache()
    user_data = sc.textFile(folder_path + "user.json").map(parse_user).cache()
    checkin_data = sc.textFile(folder_path + "checkin.json").map(parse_checkins)
    photo_data = sc.textFile(folder_path + "photo.json").map(parse_photos)
    
    photo_features = photo_data.reduceByKey(lambda a, b: a + b).collectAsMap()  # 按商家ID累加照片数，并转换为字典

    checkin_features = checkin_data.collectAsMap()  # 将结果转换为字典以便查找
    
    grouped_reviews = review_data.groupByKey().mapValues(list)

    business_features_rdd = business_data.leftOuterJoin(grouped_reviews).mapPartitions(get_business_features_partition)
    business_features = sc.broadcast(dict(business_features_rdd.collect()))

    
    user_features_rdd = user_data.mapPartitions(get_user_features_partition)
    user_features = sc.broadcast(dict(user_features_rdd.collect()))
    
    test_data_rdd = sc.textFile(test_file).filter(lambda x: not x.startswith('user_id'))
    test_data = test_data_rdd.map(lambda x: build_features(x, business_features, user_features)).collect()
    
    train_data_rdd = sc.textFile(training_data_path).zipWithIndex().filter(lambda xi: xi[1] > 0).map(lambda xi: xi[0])
    train_data = train_data_rdd.map(lambda x: build_features(x, business_features, user_features)).collect()
    
    X_train, y_train = [], []
    for user_id, business_id, features, rating in train_data:
        X_train.append(features)
        y_train.append(rating)
        
    model = XGBRegressor(max_depth=17, 
                         learning_rate=0.02,  
                         n_estimators=300,  
                         subsample=0.7,
                         colsample_bytree=0.5,
                         gamma=0.1,
                         min_child_weight=100,
                         reg_alpha=0.3,
                         reg_lambda=10)
    X_train_np = np.array(X_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32) 
    model.fit(X_train_np, y_train_np)

    X_test_np = np.array([row[2] for row in test_data], dtype=np.float32)
    model_based_preds = model.predict(X_test_np)
    
    with open(output_file, 'w') as f:
        f.write("user_id,business_id,prediction\n")
        for i in range(len(test_data)):
            record = test_data[i]
            user_id, business_id, features = record[:3] 
            
            item_based_rating = estimate_rating_item_based(user_id, business_id, user_to_business_map, business_to_user_ratings, similarity_scores_broadcast.value)
            model_based_rating = model_based_preds[i]
            
            hybrid_rating = alpha * item_based_rating + (1 - alpha) * model_based_rating  
            f.write(f"{user_id},{business_id},{hybrid_rating}\n")
            
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")        
            
    sc.stop()