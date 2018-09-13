db.tweets.find({'lang':'cs'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

