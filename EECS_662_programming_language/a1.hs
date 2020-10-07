{-# LANGUAGE GADTs,FlexibleContexts #-}

-- Imports for Monads

import Control.Monad

-- BBAE AST and Type Definitions

data TBBAE where
  TNum :: TBBAE
  TBool :: TBBAE
  deriving (Show,Eq)

data BBAE where
  Num :: Int -> BBAE
  Plus :: BBAE -> BBAE -> BBAE
  Minus :: BBAE -> BBAE -> BBAE
  Bind :: String -> BBAE -> BBAE -> BBAE
  Id :: String -> BBAE
  Boolean :: Bool -> BBAE
  And :: BBAE -> BBAE -> BBAE
  Leq :: BBAE -> BBAE -> BBAE
  IsZero :: BBAE -> BBAE
  If :: BBAE -> BBAE -> BBAE -> BBAE
  deriving (Show,Eq)

type Env = [(String,BBAE)]

type Cont = [(String,TBBAE)]

subst :: String -> BBAE -> BBAE ->BBAE
subst x v (Num n) = (Num n)
subst x v (Plus l r) = (Plus (subst x v l) (subst x v r))
subst x v (Minus l r) = (Minus (subst x v l) (subst x v r))
subst x v (Bind x' v' b') = (Bind x' (subst x v v') (subst x v b'))
subst x v (Id x') = if x == x' then v else (Id x')

evalS :: BBAE -> (Maybe BBAE)
evalS (Num x) = Just (Num x)
evalS (Boolean x) = Just (Boolean x)
evalS (Plus l r) = do {
  (Num l') <- evalS l;
  (Num r') <- evalS r;
  Just (Num (l' + r'))
}
evalS (Minus l r) = do {
  (Num l') <- evalS l;
  (Num r') <- evalS r;
  if l' >= r' then return (Num (l' - r')) else Nothing
}
evalS (Bind x a b) = do {
  (Num a') <- evalS a;
  (evalS (subst x (Num a') b))
}
evalS (Id x) = Nothing
evalS (And l r) = do {
  (Boolean l') <- evalS l;
  (Boolean r') <- evalS r;
  return (Boolean (l' && r'))
}
evalS (Leq l r) = do {
  (Num l') <- evalS l;
  (Num r') <- evalS r;
  if l' <=0 || r' <= 0 then Nothing else return (Boolean (l' <= r'))
}
evalS (IsZero x) = do {
  (Num x') <- evalS x;
  if x' < 0 then Nothing else return (Boolean (x' == 0))
}
evalS (If a b c) = do {
  (Boolean a') <- evalS a;
  if a' then (evalS b) else (evalS c)
}
--evalS _ = Nothing

evalM :: Env -> BBAE -> (Maybe BBAE)
evalM _ (Num x) = Just (Num x)
evalM e (Plus l r) = do {
  (Num l') <- evalM e l;
  (Num r') <- evalM e r;
  Just (Num (l' + r'))
}
evalM e (Minus l r) = do {
  (Num l') <- evalM e l;
  (Num r') <- evalM e r;
  if l' >= r' then return (Num (l' - r')) else Nothing
}
evalM e (Bind x a s) = do {
  a' <- evalM e a;
  (evalM ((x, a'):e) s)
}
evalM e (Id x) = do {
  v <- lookup x e;
  return v
}

--evalM _ _ = Nothing

testBBAE :: BBAE -> Bool
testBBAE x = if evalS x == evalM [] x then True else False 
--testBBAE _ = True

typeofM :: Cont -> BBAE -> (Maybe TBBAE)
typeofM _ _ = Nothing

evalT :: BBAE -> (Maybe BBAE)
evalT _ = Nothing

