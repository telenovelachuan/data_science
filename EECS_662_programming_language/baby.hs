--doubleMe x = x + x  

--doubleUs x y = doubleMe x + doubleMe y

--doubleSmallNumber x = if x > 100 then x  else x*2  

data Exp = Num Int
       | Exp Plus Exp
       | Exp :-: Exp
       | Exp :*: Exp
       | Exp :/: Exp
       deriving (Show)

eval :: Exp -> Int
eval (Num a) = a 
eval (a Plus b) = (eval a) + (eval b)
