db.tweets.find({'lang':'sl'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

