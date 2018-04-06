db.tweets.find({}).forEach(function(obj) {
    print(obj.id_str.valueOf().toString());
});
