db.tweets.find({'lang':'es'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

