db.tweets.find().forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

