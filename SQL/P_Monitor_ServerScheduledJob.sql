USE [WyethDBA]
GO
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[P_Monitor_ServerScheduledJob]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[P_Monitor_ServerScheduledJob]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE   Proc [dbo].[P_Monitor_ServerScheduledJob]	@DateID			Int,
													@ErrorCode		Int,
													@Process		Varchar(250),
													@JobDateInfo	Char(10),
													@ErrorFlag		Int,
													@ErrorMsg		Varchar(500) OUTPUT,
													@Yes			Char(1),
													@No				Char(1),
													@Zero			Int,
													@Out			Int,
													@iServerID		Int
As
/*********************************************************************/
/*		Gather Scheduled Jobs From Server(s)					     */
/*                  												 */
/* Purpose: Gather Enabled Scheduled Jobs From Servers               */
/*                                                                   */
/*  Org. Date: 02/01/2008        EPAK                                */
/*  Mod. Date: 00/00/0000        XXXX                                */
/*                                                                   */
/*********************************************************************/

SET NOCOUNT ON
/*********************************************************************/
/*																     */
/*********************************************************************/
Declare	@ServerName			sysname,
		@Command			Varchar(5000),
		@ServerID			Int,
		@Counter			Int,
		@RecCount			Int,
		@One				Int,
		@Min				Int,
		@Max				Int,
		@SeqNo				Int,
		@JobId				Uniqueidentifier,
		@JobOwner			sysname,
		@JobDesc			Varchar(5000),
		@JobEnable			SmallInt,
		@ScheduleEnable		SmallInt,
		@ScheduleFrequency	SmallInt			
			
/*********************************************************************/
/*																     */
/*********************************************************************/
Select	@ErrorFlag	= @Zero,
		@Counter	= @Zero,
		@One		= 1,
		@Out		= 3000,
		@Process	= @Process + ' - [P_Monitor_ServerScheduledJob]'

/*********************************************************************/
/*																     */
/*********************************************************************/
Create	Table #ServerInfo
 (	
	SeqNo		Int	Identity,
	ServerID	Int NOT NULL,
	ServerName	sysname	NOT NULL
 )

	Select	@ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '3001 - Unable to Create #ServerInfo Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Create	Table #ScheduledInfo
 (
	SeqNo				Int					Identity,
	JobId				Uniqueidentifier	NOT NULL,
	ServerID			Int					NOT NULL,
	JobOwner			sysname				NOT NULL,
	JobDesc				Varchar(5000)		NOT NULL,
	JobEnable			SmallInt			NOT NULL,
	ScheduleEnable		SmallInt			NOT NULL,
	ScheduleFrequency	SmallInt			NOT NULL	
 )

		Select	@ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '3002 - Unable to Create #ScheduledInfo Table'
			GOTO	Error_Trap
		 End

/*********************************************************************/
/*										                             */
/*********************************************************************/
Insert	#ServerInfo
Select	ServerID,
		ServerName
From	DatabaseServer
Where	ActiveFlag = @Yes

	Select	@ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '3003 - Unable to Insert into #ServerInfo Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Select	@Max		= Count(*),
		@Min		= @One,
		@SeqNo		= @One
From	#ServerInfo

/*********************************************************************/
/*					Remove Exisiting For Re-Run					     */
/*********************************************************************/
Delete	DatabaseServerScheduledJob 
Where	DateID = @DateID

		Select	@ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '3004 - Unable to Delete DatabaseServerScheduledJob Table'
			GOTO	Error_Trap
		 End

/*********************************************************************/
/*       Creat Procs to Extract All Scheduled Jobs From Servers      */ 
/*********************************************************************/
WHILE @SeqNo <= @Max
 Begin	-- BEGIN WHILE LOOP

	/*************************************************************/
	/*														     */
	/*************************************************************/
	Select	@ServerName = ServerName,
			@ServerID	= ServerID
	From	#ServerInfo
	Where	SeqNo = @SeqNo

	/*************************************************************/
	/*														     */
	/*************************************************************/
	SET @Command =
		'	Insert	WorkScheduledJob
			Select	'+ RTRIM(CONVERT(Char,@DateID)) +',
					ServerID,
					JobID,
	            	JobOwner,
					JobDesc,
					JobEnable,
					ScheduleEnable,
					ScheduleFrequency,
					JobStatus,
					StartRunTime,
					StartRunDate,
					LastRunDate,
					LastRunTime,
					LastRunDuration_Sec,
					JobCreateDate,
					LastUpdate
			From	[' + RTRIM(@ServerName) + '].WyethDBA.dbo.ImportScheduledJob
		'

		/*****************************************************/
		/*												     */
		/*****************************************************/
		EXEC	(@Command)

			Select	@ErrorCode	= @@ERROR,
				@RecCount	= @@ROWCOUNT
	
			IF @ErrorCode <> @Zero OR @RecCount = @Zero
			 Begin
				IF @RecCount = @Zero
				 Begin
					SET	@ErrorMsg  = 'NO Record(s) Inserted to WyethDBA.dbo.WorkScheduledJob Table for Server Name [' + RTRIM(@ServerName) + ']'
					Insert ImportProcessLog
					Values(@ServerID,Getdate(),@Process, 900, 'WARNING',@ErrorMsg)
				 End
				Else
				 Begin
					SET		@ErrorFlag = @Out
					SET		@ErrorMsg  = '3005 - Exec @Command for WorkScheduledJob Failure for Server Name [' + RTRIM(@ServerName) + ']'
					GOTO	Error_Trap
				 End
			 End

		/*****************************************************/
		/*			Current Record Check				     */
		/*****************************************************/
		IF (Select COUNT(W.DateID) From	WorkScheduledJob W INNER JOIN DateInfo D ON	W.DateID = D.DateID
			Where CONVERT(Char(10),W.LastUpdate,121) <> D.JobDateInfo And ServerID = @ServerID) > @Zero
		 Begin
			SET	@ErrorMsg  = 'OLD Records were Processed from Server Name [' + RTRIM(@ServerName) + '] & Processing Date is [' + (Select TOP 1 CONVERT(Char(10),LastUpdate,121) From WorkScheduledJob Where ServerID = @ServerID) + ']'
			Insert ImportProcessLog
			Values(@ServerID,Getdate(),@Process, 900, 'WARNING',@ErrorMsg)
		 End

	/*************************************************************/
	/*														     */
	/*************************************************************/
	Select	@SeqNo = @SeqNo + @Min

 End	-- END WHILE LOOP

/*********************************************************************/
/*																     */
/*********************************************************************/
Insert	DatabaseServerScheduledJob
(
DateID,ServerID,JobOwner,JobDesc,JobEnableCode,ScheduleEnableCode,ScheduleFreqCode,JobStatusCode,JobStartRunDate,JobStartRunTime,JobLastRunDate,JobLastRunTime,JobLastRunDuration_Sec,JobCreateDate
)

Select	@DateID,
		ServerID,
		JobOwner,
		JobDesc,
		JobEnable,
		ScheduleEnable,
		ScheduleFrequency,
		JobStatus,
		WyethDBA.dbo.fnFormatDate(StartRunDate),
		WyethDBA.dbo.fnFormatTime(StartRunTime),
		WyethDBA.dbo.fnFormatDate(LastRunDate),
		WyethDBA.dbo.fnFormatTime(LastRunTime),
		LastRunDuration_Sec,
		JobCreateDate	
From	WorkScheduledJob

	Select	@ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '3006 - Unable to Insert INTO DatabaseServerScheduledJob Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*						Gather Schedule Jobs For Reporting		     */
/*********************************************************************/
SET	@SeqNo	= 1

/*********************************************************************/
/*																     */
/*********************************************************************/
Insert	#ScheduledInfo
(
	JobId,ServerID,JobOwner,JobDesc,JobEnable,ScheduleEnable,ScheduleFrequency
)

Select	JobId,
		ServerID,
		JobOwner,
		JobDesc,
		JobEnable,
		ScheduleEnable,
		ScheduleFrequency
From	WorkScheduledJob

	Select	@ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '3007 - Unable to Insert INTO #ScheduledInfo Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*																     */
/*********************************************************************/
WHILE @SeqNo <= (Select Count(*) From #ScheduledInfo)
 Begin
	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Select	@JobId				= JobId,
			@ServerID			= ServerID,
			@JobOwner			= JobOwner,
			@JobDesc			= JobDesc,
			@JobEnable			= JobEnable,
			@ScheduleEnable		= ScheduleEnable,
			@ScheduleFrequency	= ScheduleFrequency
	From	#ScheduledInfo
	Where	SeqNo = @SeqNo

	/*************************************************************/
	/*     Insert New or Update Record(s)                        */
	/*************************************************************/
	IF ((Select Count(*) From DatabaseScheduleJobInfo Where JobID = @JobID) = @Zero)
	 Begin
		/*****************************************************/
		/*       Insert If Record(s) Does Not Exist          */
		/*****************************************************/
		Insert DatabaseScheduleJobInfo
		 (
			JobId,ServerID,JobOwner,JobDesc,JobEnableCode,ScheduleEnableCOde,ScheduleFrequencyCOde
		 )
		Select	@JobId,
				@ServerID,
				@JobOwner,
				@JobDesc,
				@JobEnable,
				@ScheduleEnable,
				@ScheduleFrequency

				Select	@ErrorCode = @@ERROR	
				IF @ErrorCode <> @Zero
				 Begin
					SET		@ErrorFlag = @Out
					SET		@ErrorMsg  = '3008 - Unable to Insert INTO DatabaseScheduleJobInfo Table'
					GOTO	Error_Trap
				 End
	 End
	ELSE
	 Begin
		/*****************************************************/
		/*    Update Existing Record Info                    */
		/*****************************************************/
		Update	DatabaseScheduleJobInfo
		Set		JobId 					= @JobId,
				ServerID 				= @ServerID,
				JobOwner				= @JobOwner,
				JobDesc					= @JobDesc,
				JobEnableCode			= @JobEnable,
				ScheduleEnableCode		= @ScheduleEnable,
				ScheduleFrequencyCOde	= @ScheduleFrequency,
				LastUpdate				= Getdate()
		Where	JobID = @JobID
		And		ServerID = @ServerID

			Select	@ErrorCode = @@ERROR	
			IF @ErrorCode <> @Zero
			 Begin
				SET		@ErrorFlag = @Out
				SET		@ErrorMsg  = '3009 - Unable to Update DatabaseScheduleJobInfo Table'
				GOTO	Error_Trap
			 End
	 End

	/*************************************************************/
	/*     Update Activity Record(s)                             */
	/*************************************************************/
	IF @JobEnable = @Zero OR @ScheduleEnable = @Zero
	 Begin
		Update	DatabaseScheduleJobInfo
		Set		Active	 = @No
		Where	JobID	 = @JobID
		And		ServerID = @ServerID

			Select	@ErrorCode = @@ERROR	
			IF @ErrorCode <> @Zero
			 Begin
				SET		@ErrorFlag = @Out
				SET		@ErrorMsg  = '3010 - Unable to Update DatabaseScheduleJobInfo Table Active Indicator to (No)'
				GOTO	Error_Trap
			 End
	 End
	Else
	 Begin
		Update	DatabaseScheduleJobInfo
		Set		Active = @Yes
		Where	JobID = @JobID
		And		ServerID = @ServerID

			Select	@ErrorCode = @@ERROR	
			IF @ErrorCode <> @Zero
			 Begin
				SET		@ErrorFlag = @Out
				SET		@ErrorMsg  = '3010 - Unable to Update DatabaseScheduleJobInfo Table Active Indicator to (Yes)'
				GOTO	Error_Trap
			 End
	 End

	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Select	@SeqNo = @SeqNo + @Min
 End

/*********************************************************************/
/*																     */
/*********************************************************************/
IF (Select Count(*) From DatabaseScheduleJobInfo Where JobId NOT IN (Select JobID From #ScheduledInfo)And Active = @Yes) <> @Zero
 Begin
	Update	DatabaseScheduleJobInfo
	Set		Active = @No,
			LastUpdate = Getdate()
	Where	JobId NOT IN (Select JobID From #ScheduledInfo)
	And 	Active = @Yes

		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '3011 - Unable to Update DatabaseScheduleJobInfo Table for Active Indicator For Inactive Record'
			GOTO	Error_Trap
		 End
 End

/*********************************************************************/
/*			House Cleaning										     */
/*********************************************************************/
Drop Table #ServerInfo
Drop Table #ScheduledInfo

/*********************************************************************/
/*   Update Unknow Date & Time With With Job Start Date & Time       */
/*********************************************************************/
Update	DatabaseServerScheduledJob
SET		JobLastRunDate = JobStartRunDate,
		JobLastRunTime = JobStartRunTime
Where	JobLastRunDate = 'Unknown'
And		DateID = @DateID

		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '3012 - Unable to Update UNKNOWN Date & Time on DatabaseScheduleJobInfo Table'
			GOTO	Error_Trap
		 End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Insert ImportProcessLog
Values(@iServerID,Getdate(),@Process, @@ERROR, 'SUCCESS','***** COMPLETED - Gather & Distribute Database Scheduled Jobs from Remote Servers')

/*********************************************************************/
/*																     */
/*********************************************************************/
Error_Trap:
IF @ErrorFlag <> @Zero
 Begin
	Insert ImportProcessLog
	Values(@iServerID,Getdate(),@Process, @ErrorFlag, 'FAILURE',@ErrorMsg)
	RETURN @ErrorFlag
 End

/*********************************************************************/
/*																     */
/*********************************************************************/
SET NOCOUNT OFF