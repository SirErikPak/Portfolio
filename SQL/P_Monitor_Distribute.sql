USE [WyethDBA]
GO
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[P_Monitor_Distribute]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[P_Monitor_Distribute]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE Proc [dbo].[P_Monitor_Distribute]	@DateID			Int,
											@Process		Varchar(250),
											@ErrorCode		Int,
											@ErrorFlag		Int,
											@ErrorMsg		Varchar(500) OUTPUT,
											@Zero			Int,
											@Yes			Char(1),
											@Out			Int,
											@iServerID		Int
As
/*********************************************************************/
/*		Populate DatabaseInfo Table for Reporting				     */
/*                  											     */
/* Purpose: Populate DatabaseInfo Table for Reporting			     */
/*                                                                   */
/*  Org. Date: 02/01/2007        EPAK                                */
/*  Mod. Date: 00/00/0000        XXXX                                */
/*                                                                   */
/*********************************************************************/

SET NOCOUNT ON
/*********************************************************************/
/*																     */
/*********************************************************************/
Declare	@PriorDateID			Int,
		@DatabaseNameID			Int,
		@ServerID				Int,
		@DiskSizeThresholdID	Int,
		@DiskSizeThresholdMax	Int,
		@DiskSizeThresholdMin	Int,
		@DatabaseSize			Decimal(18,7),
		@DataSize				Decimal(18,7),
		@LogSize				Decimal(18,7),
		@SeqNo					Int,
		@Counter				Int,
		@One					Int,
		@Two					Int

/*********************************************************************/
/*																     */
/*********************************************************************/
Select	@SeqNo		= 1,
		@One 		= 1,
		@Two 		= 2,
		@ErrorFlag	= 0,
		@Out		= 5000,
		@Process	= @Process + ' - [P_Monitor_Distribute]'

/*********************************************************************/
/*																     */
/*********************************************************************/
Create	Table #DataBaseInfo
(
	SeqNo			Int	Identity ,
	ServerID		Int	NOT NULL ,
	DatabaseNameID	Int	NOT NULL ,
	DatabaseSize	Float	NOT NULL ,
	DataSize		Float	NOT NULL ,
	LogSize			Float	NOT NULL
)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '5001 - Unable to Create #DataBaseInfo Temp Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
Create	Table #SpaceThreshold
(
	SeqNo					Int	Identity ,
	DiskSizeThresholdID		Int	NOT NULL ,
	DiskSizeThresholdMax_MB	Int	NOT NULL ,
	DiskSizeThresholdMIn_MB	Int	NOT NULL
)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '5001 - Unable to Create #SpaceThreshold Temp Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Select	@PriorDateID = (Select Distinct DateID From DatabaseServerDiskInfo Where DateID = (Select MAX(DateID) From DatabaseServerDiskInfo))

/*********************************************************************/
/*																     */
/*********************************************************************/
Delete	DatabaseInfo
Where	DateID = @DateID

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '5002 - Unable to Delete Existing DateID From DatabaseInfo Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Insert	DatabaseInfo
(
	DateID,ServerID,DatabaseNameID,DatabaseOwnerID,DatabaseSize_MB,DataSize_MB,DataPercentUsed,LogSize_MB,
   	LogPercentUsed,DatabaseStatusID,DatabaseUpdateabilityID,DatabaseUserAccessID,DatabaseRecoveryID,
	DatabaseCollationID,DatabaseSortOrder,DatabaseCreateDate
)

Select	@DateID,
		WDI.ServerID,
		DN.DatabaseNameID,
		DO.DatabaseOwnerID,
		WDI.DatabaseSize_MB,
		WDI.DatabaseDataSize_MB,
		WDI.DatabasePercentUsed,
		WDI.DatabaseLogSize_MB,
		WDI.DatabaseLogPercentUsed,
		DS.DatabaseStatusID,
		DU.DatabaseUpdateabilityID,
		DA.DatabaseUserAccessID,
		DR.DatabaseRecoveryID,
		DC.DatabaseCollationID,
		CASE
		 WHEN ISNUMERIC(RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('SQLSortOrder',WDI.DatabaseStatus)) + @One),@Two))) = 0 THEN 0
		 WHEN RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('SQLSortOrder',WDI.DatabaseStatus)) + @One),@Two)) = '0,' THEN 0
		 ELSE CONVERT(SmallInt,RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('SQLSortOrder',WDI.DatabaseStatus)) + @One),@Two)))
		END  As [Database SQL Sort Order],
		WDI.DatabaseCreateDate

From	WorkDatabaseInfo WDI INNER JOIN DatabaseName DN
ON		WDI.DatabaseName = DN.DatabaseName
		INNER JOIN DatabaseOwner DO
ON		WDI.DatabaseOwner = DO.DatabaseOwner
		INNER JOIN DatabaseStatus DS
ON		RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,@One) + @One),(CHARINDEX(',',WDI.DatabaseStatus,@One) - @One) - (CHARINDEX('=',WDI.DatabaseStatus,@One)))) = DS.DatabaseStatus
		INNER JOIN DatabaseUpdateability DU
ON		RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,(CHARINDEX('Updateability',WDI.DatabaseStatus))) + @One),((CHARINDEX(',',WDI.DatabaseStatus,(CHARINDEX('Updateability',WDI.DatabaseStatus)))) - (CHARINDEX('=',WDI.DatabaseStatus,(CHARINDEX('Updateability',WDI.DatabaseStatus))) + @One)))) = DU.DatabaseUpdateability
		INNER JOIN DatabaseUserAccess DA
ON		RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('UserAccess',WDI.DatabaseStatus)) + @One),((CHARINDEX(',',WDI.DatabaseStatus,CHARINDEX('UserAccess',WDI.DatabaseStatus))) - (CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('UserAccess',WDI.DatabaseStatus)) + @One)))) = DA.DatabaseUserAccess
		INNER JOIN DatabaseRecovery DR
ON		RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('Recovery',WDI.DatabaseStatus)) + @One),((CHARINDEX(',',WDI.DatabaseStatus,CHARINDEX('Recovery',WDI.DatabaseStatus))) - (CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('Recovery',WDI.DatabaseStatus)) + @One)))) = DR.DatabaseRecovery
		INNER JOIN DatabaseCollation DC
ON		RTRIM(SUBSTRING(WDI.DatabaseStatus,(CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('Collation',WDI.DatabaseStatus)) + @One),((CHARINDEX(',',WDI.DatabaseStatus,CHARINDEX('Collation',WDI.DatabaseStatus))) - (CHARINDEX('=',WDI.DatabaseStatus,CHARINDEX('Collation',WDI.DatabaseStatus)) + @One)))) = DC.DatabaseCollation

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '5003 - Unable to Insert DatabaseInfo Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*			Delete Existing DateID from DatabaseFile Table		     */
/*********************************************************************/
Delete	Databasefile
Where	DateID = @DateID

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5004 - Delete Existing DateID from DatabaseFile Table Failure'
			GOTO	Error_Trap
		 End

/*********************************************************************/
/*					Insert ONLY Database File Info					 */
/*********************************************************************/
Insert	Databasefile
 (
	DateID,ServerID,DatabaseNameID,DatabaseLogicalName,
	DatabaseFileLocation,Size_MB,GrowthLimit_MB,GrowthSize_MB,
	GrowthPercent,FileType
 )
Select	WDF.DateID,WDF.ServerID,DN.DatabaseNameID,WDF.DatabaseLogicalName,
		WDF.DatabaseFileLocation,WDF.Size_MB,WDF.GrowthLimit_MB,WDF.GrowthSize_MB,
		WDF.GrowthPercent,WDF.FileType
From	WorkDatabaseFile WDF INNER JOIN DatabaseName DN
ON		WDF.DatabaseName = DN.DatabaseName

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5005 - Insert Databasefile Table Failure'
			GOTO	Error_Trap
		 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Delete	DatabaseServerDiskInfo
Where	DateID = @DateID

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '5006 - Unable to Delete Existing DateID From DatabaseServerDiskInfo Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Insert	#SpaceThreshold 
 (
	DiskSizeThresholdID,DiskSizeThresholdMin_MB,DiskSizeThresholdMax_MB
 )
Select	DiskSizeThresholdID,
		DiskSizeThresholdMin_GB,
		DiskSizeThresholdMax_GB
From	DatabaseDiskSizeThreshold

/*********************************************************************/
/*					Re-Set @SeqNo to 1							     */
/*********************************************************************/
Select	@SeqNo						= 1,
		@Counter					= Count(*)
From	DatabaseDiskSizeThreshold

/*********************************************************************/
/*																     */
/*********************************************************************/
WHILE @SeqNo <= (@Counter)

 BEGIN
	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Select	@DiskSizeThresholdID	= DiskSizeThresholdID,
			@DiskSizeThresholdMin	= DiskSizeThresholdMin_MB,
			@DiskSizeThresholdMax	= DiskSizeThresholdMax_MB
	From	#SpaceThreshold
	Where	SeqNo = @SeqNo

	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Insert	DatabaseServerDiskInfo
	(
		DateID,
		ServerID,
		DriveLetter,
		TotalDiskSpace_MB,
		FreeDiskSpace_MB,
		DiskSizeThresholdID,
		FreeDiskPercentage
	)

	Select	@DateID,
			ServerID,
			DriveLetter,
			TotalDiskSpace_MB,
			FreeDiskSpace_MB,
			@DiskSizeThresholdID,
			CONVERT(Float,FreeDiskSpace_MB)/CONVERT(Float,TotalDiskSpace_MB) As FreeDiskPercentage

	From	WorkDiskInfo
	Where	TotalDiskSpace_MB >  @DiskSizeThresholdMin * 1000		---- MB to GB Convert
	And		TotalDiskSpace_MB <= @DiskSizeThresholdMax * 1000		---- MB to GB Convert

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5007 - Unable to Insert DatabaseServerDiskInfo Table'
			GOTO	Error_Trap
		 End

	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Select	@SeqNo	= @SeqNo + @One

 END

/*********************************************************************/
/*																     */
/*********************************************************************/
Drop Table #SpaceThreshold

/*********************************************************************/
/*	    Delete Existing DateID from DatabaseFileSummary Table		 */
/*********************************************************************/
Delete	DatabaseFileSummary
Where	DateID = @DateID

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5008 - Delete Existing DateID from DatabaseFileSummary Table Failure'
			GOTO	Error_Trap
		 End

/*********************************************************************/
/*			Insert ONLY Database File Summary Info					 */
/*********************************************************************/
Insert	DatabaseFileSummary
Select	DI.DateID,
		DS.SiteID,
		DI.ServerID,
		DI.DatabaseNameID,
		UPPER(Final.FileType) As FileType,
		CASE
		 WHEN Final.FileType = 'Data Only' THEN DI.DataSize_MB
		 ELSE LogSize_MB
		END As Data_Log_Size_MB,
		CASE
		 WHEN Final.FileType = 'Data Only' THEN Final.GrowthLimit_MB
		 ELSE Final.GrowthLimit_MB
		END As GrowthLimit_MB,
		CASE
		 WHEN Final.FileType = 'Data Only' THEN Final.GrowthSize_MB
		 ELSE Final.GrowthSize_MB
		END As GrowthSize_MB,
		CASE
		 WHEN Final.FileType = 'Data Only' THEN Final.GrowthPercent
		 ELSE Final.GrowthPercent
		END As GrowthPercent,
		0 As Data_Log_Free_MB,
		CASE
		 WHEN Final.FileType = 'Data Only' THEN (DI.DataPercentUsed/100)
		 ELSE (DI.LogPercentUsed/100)
		END As PercentUsed,
		0,
		User_Name(),
		Getdate()
From	DatabaseInfo DI INNER JOIN DatabaseServer DS
ON		DI.ServerID = DS.ServerID
		INNER JOIN
 (
	Select	DateID,
			ServerID,
			DatabaseNameID,
			SUM(ISNULL(Size_MB,0)) As Size_MB,
			SUM(ISNULL(GrowthLimit_MB,0)) As GrowthLimit_MB,
			SUM(ISNULL(GrowthSize_MB,0)) As GrowthSize_MB,
			SUM(ISNULL(GrowthPercent,0)) As GrowthPercent,
			FileType
	From	DatabaseFile
	Where	DateID = @DateID
	Group By DateID,ServerID,DatabaseNameID,FileType
 ) As Final
ON		DI.DateID = Final.DateID
And		DI.ServerID = Final.ServerID
And		DI.DatabaseNameID = Final.DatabaseNameID
Where	DI.DateID = @DateID

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5009 - Insert New DateID INTO DatabaseFileSummary Table Failure'
			GOTO	Error_Trap
		 End

/*********************************************************************/
Update	DatabaseFileSummary
SET		GrowthLimit_MB = ISNULL(DF.GrowthLimit_MB,0)
From	DatabaseFile DF INNER JOIN DatabaseFileSummary FS
ON		DF.DateID = FS.DateID
And		DF.ServerID = FS.ServerID
And		DF.DatabaseNameID = FS.DatabaseNameID
And		UPPER(DF.FileType) = UPPER(FS.FileType)		
Where	DF.GrowthLimit_MB IS NULL
And		FS.GrowthLimit_MB > 0
And		DF.DateID = @DateID

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5010 - Update Growth Limit on DatabaseFileSummary Table Failure'
			GOTO	Error_Trap
		 End

/*********************************************************************/
Update	DatabaseFileSummary
SET		Data_log_Free_MB = GrowthLimit_MB - (Data_Log_Size_MB * PercentUsed),
		PercentFree = (1.00 - (Data_Log_Size_MB * PercentUsed)/GrowthLimit_MB)
Where	GrowthLimit_MB > 0
And		DateID = @DateID

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5011 - Update Free Sapce & Percentage on DatabaseFileSummary Table Failure'
			GOTO	Error_Trap
		 End

/*********************************************************************/
Update	DatabaseFileSummary
SET		PercentFree = 1
From	DatabaseFileSummary
Where	GrowthLimit_MB = 0
And		DateID = @DateID	

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5012 - Update Percent Free Sapce on DatabaseFileSummary Table Failure'
			GOTO	Error_Trap
		 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Insert	#DataBaseInfo
	(
		ServerID,
		DatabaseNameID,
		DatabaseSize,
		DataSize,
		LogSize
	)

Select	ServerID,
		DatabaseNameID,
		DatabaseSize_MB,
		DataSize_MB,
		LogSize_MB
From	DatabaseInfo
Where	DateID = ISNULL(@PriorDateID,0)
Order	By DateID, ServerID, DatabaseNameID

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '5013 - Unable to Insert #DataBaseInfo Temp Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*					Re-Set @SeqNo to 1							     */
/*********************************************************************/
Select	@SeqNo		= 1,
		@Counter	= Count(*)
From	#DataBaseInfo

/*********************************************************************/
/*																     */
/*********************************************************************/
WHILE @SeqNo <= (@Counter)

 BEGIN
	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Select	@ServerID		= ServerID,
			@DatabaseNameID	= DatabaseNameID,
			@DatabaseSize	= DatabaseSize,
			@DataSize		= DataSize,
			@LogSize		= LogSize
	From	#DataBaseInfo
	Where	SeqNo = @SeqNo

	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Update	DatabaseInfo
	Set		DatabaseSizeChange_MB	= (DatabaseSize_MB - ISNULL(@DatabaseSize,0)),
			DataSizeChange_MB		= (DataSize_MB - ISNULL(@DataSize,0)),
			LogSizeChange_MB		= (LogSize_MB - ISNULL(@LogSize,0))
	Where	DateID = ISNULL(@DateID,0)
	And		ServerID = ISNULL(@ServerID,0)
	And		DatabaseNameID = ISNULL(@DatabaseNameID,0)

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '5014 - Unable to Update DatabaseInfo Table'
			GOTO	Error_Trap
		 End

	/*************************************************************/
	/*                                                           */
	/*************************************************************/
	Select	@SeqNo 	 = @SeqNo + @One

 END

/*********************************************************************/
/*																     */
/*********************************************************************/
Drop Table #DataBaseInfo

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Insert ImportProcessLog
Values(@iServerID,Getdate(),@Process, @@ERROR, 'SUCCESS','***** COMPLETED - Distribute Data from Scheduled Jobs & Disk Info & Logins From Remote Servers to Proper Tables')

/*********************************************************************/
/*																     */
/*********************************************************************/
Error_Trap:
IF @ErrorFlag <> @Zero
 Begin
	Insert ImportProcessLog
	Values(@iServerID,Getdate(),@Process, @ErrorCode, 'FAILURE',@ErrorMsg)
	RETURN @ErrorFlag
 End

/*********************************************************************/
/*																     */
/*********************************************************************/
SET NOCOUNT OFF
