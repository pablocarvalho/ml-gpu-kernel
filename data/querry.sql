--Creates a view for concurrent experiment

CREATE TEMP VIEW view_kernel_concurrency_experiment

AS
	select k1.codeName as k1_name, k2.codeName as k2_name, (k1.gridX + k1.gridY + k1.gridZ) as k1_blocks_per_grid, (k2.gridX + k2.gridY + k2.gridZ) as k2_blocks_per_grid,
	(k1.blockX + k1.blockY + k1.blockZ) as k1_threads_per_block, (k2.blockX + k2.blockY + k2.blockZ) as k2_threads_per_block, k1.registersPerThread as k1_registers,  
	k2.registersPerThread as k2_registers, k1.staticSharedMemory as k1_shared_mem_B, k2.staticSharedMemory as k2_shared_mem_B, SUM( case when concurrency_duration > 0 then 1 else 0 end) as valid_attempts,
	case when (SUM( case when concurrency_duration > 0 then 1 else 0 end)) >= 4 then 1 else 0 end as classification from
		concurrent_experiment as ce
			inner join 
		kernels as k1
			on ce.id_kernel1 == k1._id_ 
			inner join
		kernels as k2
			on ce.id_kernel2 == k2._id_ 
			inner join
		application as app_1
			on k1.application == app_1._id_	
			inner join
		application as app_2		
			on k2.application == app_2._id_			
		group by id_experiment;

-- Outputs created view

.headers on
.mode csv
.tesla_p100_CE_classification.csv
select * from view_kernel_concurrency_experiment;


-- create a view for interference experiment

CREATE TEMP VIEW view_application_concurrency_experiment2	
as
select k1.codeName as k1_name, k2.codeName as k2_name, (k1.gridX + k1.gridY + k1.gridZ) as k1_blocks_per_grid, (k2.gridX + k2.gridY + k2.gridZ) as k2_blocks_per_grid,
	(k1.blockX + k1.blockY + k1.blockZ) as k1_threads_per_block, (k2.blockX + k2.blockY + k2.blockZ) as k2_threads_per_block, k1.registersPerThread as k1_registers,  
	k2.registersPerThread as k2_registers, k1.staticSharedMemory as k1_shared_mem_B, k2.staticSharedMemory as k2_shared_mem_B, SUM( case when concurrency_duration > 0 then 1 else 0 end) as valid_attempts,	
	case when (SUM( case when concurrency_duration > 0 then 1 else 0 end)) >= 4 then 1 else 0 end as classification, 
	((AVG(case when concurrency_duration > 0 then ce.k1_end - ce.k1_start else null end) - k1.avgtimefirst)/AVG(case when concurrency_duration > 0 then concurrency_duration else null end)) as k1_ce, 
	((AVG(case when concurrency_duration > 0 then ce.k2_end - ce.k2_start else null end) - k2.avgtimefirst)/AVG(case when concurrency_duration > 0 then concurrency_duration else null end)) as k2_ce,
	 AVG( (case when concurrency_duration > 0 then k1.avgtimefirst + k2.avgtimefirst else null end) )/AVG(case when concurrency_duration > 0 then ce.total_duration else null end) as speedup from
		concurrent_experiment as ce
			inner join 
		kernels as k1
			on ce.id_kernel1 == k1._id_
			inner join
		kernels as k2
			on ce.id_kernel2 == k2._id_ 		
		group by id_experiment;

-- outputs a view for inteference experiment
.headers on
.mode csv
.tesla_p100_CE_classification.csv
select k1_name, k2_name,k1_blocks_per_grid,k2_blocks_per_grid,k1_threads_per_block,k2_threads_per_block,k1_registers,k2_registers,k1_shared_mem_B,k2_shared_mem_B, case when speedup > 1 then 1 else 0 end as classification from view_application_concurrency_experiment2 where valid_attempts >= 4;