-- Create the users table
CREATE TABLE IF NOT EXISTS `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `email` varchar(255) NOT NULL UNIQUE,
  `name` varchar(255),
  `country` enum ('US', 'CO','TN') NOT NULL DEFAULT 'US',
  PRIMARY KEY (`id`)
)
