export function nextStackOpen({ isActive, wasActive, currentOpen }) {
  if (isActive && !wasActive) return true;
  if (!isActive) return false;
  return currentOpen;
}
