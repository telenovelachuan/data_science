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

-- subst function
subst :: String -> FBAE -> FBAE -> FBAE
subst x v (Num n) = (Num n)
subst x v (Plus l r) = (Plus (subst x v l) (subst x v r))
subst x v (Minus l r) = (Minus (subst x v l) (subst x v r))
subst x v (Mult l r) = (Mult (subst x v l) (subst x v r))
subst x v (Div l r) = (Div (subst x v l) (subst x v r))

subst x v (Bind x' v' b') = if x==x' then (Bind x' (subst x v v') b') else (Bind x' (subst x v v') (subst x v b'))
subst x v (Id x') = if x == x' then v else (Id x')
subst x v (Lambda x' t' b') = (Lambda x' t' (subst x v b'))
subst x v (App f' a') = (App (subst x v f') (subst x v a') )
subst x v (Boolean b) = (Boolean b)
subst x v (And l r) = (And (subst x v l) (subst x v r))
subst x v (Or l r) = (Or (subst x v l) (subst x v r))
subst x v (Leq l r) = (Leq (subst x v l) (subst x v r))
subst x v (IsZero a) = (IsZero (subst x v a))
subst x v (If c t f) = (If (subst x v c) (subst x v t) (subst x v f))
subst x v (Fix f) = (Fix (subst x v f))


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
  (evalM ((i, a'):e') b)
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
  (evalM e' (subst i (Fix f) b))
}


--evalM _ _ = Nothing

-- Type inference function

type Cont = [(String,TFBAE)]

typeofM :: Cont -> FBAE -> (Maybe TFBAE)
typeofM c (Num x) = Just TNum
typeofM c (Boolean b) = Just TBool
typeofM c (Plus l r) = do {
  TNum <- (typeofM c l);
  TNum <- (typeofM c r);
  return TNum;
}
typeofM c (Minus l r) = do {
  TNum <- (typeofM c l);
  TNum <- (typeofM c r);
  return TNum;
}
typeofM c (Mult l r) = do {
  TNum <- (typeofM c l);
  TNum <- (typeofM c r);
  return TNum;
}
typeofM c (Div l r) = do {
  TNum <- (typeofM c l);
  TNum <- (typeofM c r);
  return TNum;
}
typeofM c (And l r) = do {
  TBool <- (typeofM c l);
  TBool <- (typeofM c r);
  return TBool;
}
typeofM c (Or l r) = do {
  TBool <- (typeofM c l);
  TBool <- (typeofM c r);
  return TBool;
}
typeofM c (Leq l r) = do {
  TBool <- (typeofM c l);
  TBool <- (typeofM c r);
  return TBool;
}
typeofM c (IsZero x) = do {
  TBool <- (typeofM c x);
  return TBool;
}
typeofM c (If x y z) = do {
  TBool <- (typeofM c x);
  y' <- (typeofM c y);
  z' <- (typeofM c z);
  if y'==z' then return y' else Nothing
}
typeofM c (Bind i v b) = do {
  v' <- (typeofM c v);
  (typeofM ((i, v'):c) b)
}
typeofM c (Lambda i t b) = do {
  r <- (typeofM ((i, t):c) b);
  return (t:->:r) 
}
typeofM c (App f a) = do {
  a' <- (typeofM c a);
  (d:->:r) <- (typeofM c f);
  if d==a' then return r else Nothing 
}
typeofM c (Id i) = lookup i c
typeofM c (Fix f) = do {
  (d:->:r) <- (typeofM c f);
  return r
}
--typeofM _ _ = Nothing


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
