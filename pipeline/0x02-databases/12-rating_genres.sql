-- Display genres by their rating
SELECT tv_genres.name AS name, sum(tv_show_ratings.rate) AS rating
FROM tv_show_genres JOIN tv_show_ratings ON tv_show_genres.show_id = tv_show_ratings.show_id
JOIN tv_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP  BY tv_genres.name
ORDER BY rating DESC, name;
