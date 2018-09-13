db.tweets.find({'lang':'fr'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

