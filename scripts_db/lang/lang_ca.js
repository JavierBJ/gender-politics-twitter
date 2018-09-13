db.tweets.find({'lang':'ca'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

