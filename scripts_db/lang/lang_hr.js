db.tweets.find({'lang':'hr'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

