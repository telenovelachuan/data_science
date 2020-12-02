{-# LANGUAGE GADTs #-}

-- import Control.Monad

-- Calculator language extended with an environment to hold defined variables

data TFBAE where
  TNum :: TFBAE
  TBool :: TFBAE
  (:->:) :: TFBAE -> TFBAE -> TFBAE
  deriving (Show,Eq)

data FBAE where
  Num :: Int -> FBAE
  Plus :: FBAE -> FBAE -> FBAE
  Minus :: FBAE -> FBAE -> FBAE
  Mult :: FBAE -> FBAE -> FBAE
  Div :: FBAE -> FBAE -> FBAE
  Bind :: String -> FBAE -> FBAE -> FBAE
  Lambda :: String -> TFBAE -> FBAE -> FBAE
  App :: FBAE -> FBAE -> FBAE
  Id :: String -> FBAE
  Boolean :: Bool -> FBAE
  And :: FBAE -> FBAE -> FBAE
  Or :: FBAE -> FBAE -> FBAE
  Leq :: FBAE -> FBAE -> FBAE
  IsZero :: FBAE -> FBAE
  If :: FBAE -> FBAE -> FBAE -> FBAE
  Fix :: FBAE -> FBAE
  deriving (Show,Eq)

-- Value defintion for statically scoped eval

data FBAEVal where
  NumV :: Int -> FBAEVal
  BooleanV :: Bool -> FBAEVal
  ClosureV :: String -> FBAE -> Env -> FBAEVal
  deriving (Show,Eq)

-- Enviornment for statically scoped eval

type Env = [(String,FBAEVal)]

-- Statically scoped eval
         
evalM :: Env -> FBAE -> (Maybe FBAEVal)

evalM e (Num x) = Just (NumV x)
evalM e (Plus l r) = do {
  (NumV l') <- (evalM e l);
  (NumV r') <- (evalM e r);
  return (NumV (l' + r'))
}
evalM e (Minus l r) = do {
  (NumV l') <- (evalM e l);
  (NumV r') <- (evalM e r);
  return (NumV (l' - r'))
}
evalM e (Mult l r) = do {
  (NumV l') <- (evalM e l);
  (NumV r') <- (evalM e r);
  return (NumV (l' * r'))
}
evalM e (Div l r) = do {
  (NumV l') <- (evalM e l);
  (NumV r') <- (evalM e r);
  return (NumV (div l' r'))
}
evalM e (Bind i v b) = do {
  v' <- (evalM e v);
  evalM ((i, v'):e) b
}
evalM e (Lambda i t b) = do {
  Just (ClosureV i b e)
}
evalM e (App f a) = do {
  (ClosureV i b e') <- (evalM e f);
  a' <- (evalM e a);
  (evalM (i, a'):e' b)
}
evalM e (Id x) = (lookup x e)
evalM e (Boolean b) = Just (BooleanV b)
evalM e (And l r) = do {
  (BooleanV l') <- (evalM e l);
  (BooleanV r') <- (evalM e r);
  return (BooleanV (l' && r'))
}
evalM e (Or l r) = do {
  (BooleanV l') <- (evalM e l);
  (BooleanV r') <- (evalM e r);
  return (BooleanV (l' || r'))
}
evalM e (Leq l r) = do {
  (BooleanV l') <- (evalM e l);
  (BooleanV r') <- (evalM e r);
  return (BooleanV (l' <= r'))
}
evalM e (IsZero x) = do {
  (NumV x') <- (evalM e x);
  return (BooleanV (x' == 0))
}
evalM e (If a b c) = do {
  (BooleanV a') <- (evalM e a);
  if a' then (evalM e b) else (evalM e c)
}
evalM e (Fix f) = do {
  (ClosureV i b e') <- (evalM e f);
  
}


--evalM _ _ = Nothing

-- Type inference function

type Cont = [(String,TFBAE)]

typeofM :: Cont -> FBAE -> (Maybe TFBAE)
typeofM _ _ = Nothing


-- Interpreter

interp :: FBAE -> (Maybe FBAEVal)
interp _ = Nothing

-- Factorial function for testing evalM and typeofM.  the type of test1 should
-- be TNum and the result of evaluating test1`should be (NumV 6).  Remember
-- that Just is used to return both the value and type.

test1 = (Bind "f" (Lambda "g" ((:->:) TNum TNum)
                    (Lambda "x" TNum (If (IsZero (Id "x")) (Num 1)
                                         (Mult (Id "x")
                                               (App (Id "g")
                                                    (Minus (Id "x")
                                                           (Num 1)))))))
         (App (Fix (Id "f")) (Num 3)))
