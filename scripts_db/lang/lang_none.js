db.tweets.find({'lang':'none'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

