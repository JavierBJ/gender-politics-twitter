db.tweets.find({'lang':'de'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

