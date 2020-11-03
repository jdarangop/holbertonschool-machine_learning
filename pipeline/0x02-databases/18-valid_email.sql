-- Creates a trigger that resets the attribute valid_email only when the email has been changed.
DELIMITER $$
CREATE TRIGGER `email_changed`
  BEFORE UPDATE
  ON users FOR EACH ROW
BEGIN
  IF STRCMP(OLD.email, NEW.email) != 0 THEN
    SET NEW.valid_email = IF (NEW.valid_email, 0, 1);
  END IF;
END $$
DELIMITER ;
