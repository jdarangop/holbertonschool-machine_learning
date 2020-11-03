-- Display number of shows by each genre
SELECT tv_genres.name as genre, COUNT(tv_show_genres.genre_id) AS number_of_shows
FROM tv_genres JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP BY genre ORDER BY number_of_shows DESC, genre WHERE number_of_shows >= 1;
