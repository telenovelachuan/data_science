--doubleMe x = x + x  

--doubleUs x y = doubleMe x + doubleMe y

--doubleSmallNumber x = if x > 100 then x  else x*2  

liftNum f num1 num2 = f num1 num2

test = do {l' <- (1);
                       r' <- (2);
                       Just l' + r'}

