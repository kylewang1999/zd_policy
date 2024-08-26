function [pos, isterminal,direction] = explosionEvent(~, x, bound)
pos = norm(x)-bound;
isterminal = 1;
direction = 1;
end