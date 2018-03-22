db.tweets.find({"msg":"tweet"}).forEach(function(obj) {
    print(obj.id_str.valueOf().toString());
});
