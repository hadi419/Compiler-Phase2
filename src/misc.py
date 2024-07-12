def expect_or(cond, msg):
  if cond:
    return
  
  exit(msg)