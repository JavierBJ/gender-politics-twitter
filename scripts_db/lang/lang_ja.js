db.tweets.find({'lang':'ja'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

