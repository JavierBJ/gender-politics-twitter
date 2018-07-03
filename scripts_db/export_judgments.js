db.tweets.find({'is_hostile':{'$in':[1,-1,0.5,2]}}).forEach(function(obj) {
    print(obj.hostile_1.valueOf().toString()+','+obj.hostile_2.valueOf().toString());
});
