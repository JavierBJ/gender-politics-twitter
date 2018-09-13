db.tweets.find({'lang':'it'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

