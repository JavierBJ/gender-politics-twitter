db.tweets.find({'lang':'en'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

