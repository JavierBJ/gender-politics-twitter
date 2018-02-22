result = db.users.aggregate(
  {'$group': {_id:'$screen_name', count:{$sum:1}}},
  {'$match': {'_id': {'$ne': null }, 'count': {'$ne':1}}});

if (!result.hasNext()) {
  print('Pass.');
} else {
  print('Fail.');
}

