db.tweets.find({'lang':'gl'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

