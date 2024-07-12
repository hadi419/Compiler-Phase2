#include <stdio.h>

int recursiveFib(int n);

int internalRecursiveFib(int n)
{
  if (n < 2)
    return n;

  return internalRecursiveFib(n - 1) + internalRecursiveFib(n - 2);
}


int main()
{
  for (int i = 0; i <= 10; i++)
  {
    int f1 = recursiveFib(i);
    int f2 = internalRecursiveFib(i);

    if (f1 != f2)
      return 1;

    printf("recursiveFib(%d) -> %d\n", i, f1);
  }

  return 0;
}
