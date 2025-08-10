USE machine_intelligence;

truncate active_neurons;
truncate connections;
truncate coordinates;
delete from neurons;
select * from neurons;
select * from coordinates where neuron_id = 40;
select * from connections;
select * from active_neurons;

-- SELECT '{"x":0.11,"y":0.21,"r":1,"g":0,"b":0}' as point_str, (
		-- SELECT CONCAT(neuron_id, '|', distance) as neuron_distance
		-- FROM (
				SELECT neuron_id, dimension_id, SQRT(SUM(
					CASE
                    WHEN dimension_id = 1 THEN POW( val - 0.11, 2)
                    WHEN dimension_id = 2 THEN POW( val - 0.21, 2)
                    WHEN dimension_id = 3 THEN POW(val - 1, 2)
                    WHEN dimension_id = 4 THEN POW(val - 0, 2)
                    WHEN dimension_id = 5 THEN POW(val - 0, 2)
                    END)) AS distance
				FROM coordinates
				WHERE neuron_id IN (39)
				AND dimension_id IN (1,2,3,4,5);
				-- GROUP BY neuron_id
				-- HAVING distance <= 1e-8
				-- ORDER BY distance
				-- LIMIT 1;
		-- ) q;
-- ) as neuron_distance;

SELECT s.neuron_id as source_id, t.neuron_id as target_id, 1 / (1 + t.age) as strength -- as the age difference increases, strength decreases
FROM active_neurons s
CROSS JOIN active_neurons t
WHERE s.level = 0 -- get the active neurons in the given level
AND s.age = 0 -- reinforcing connections for the newly activated neurons only
AND t.level = s.level -- reinforcing connections only within the same level
AND (t.neuron_id != s.neuron_id OR t.age != s.age); -- if it's the same neuron, it's gotta be an older one
