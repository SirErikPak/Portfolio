if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[P_GECS_Determine_Primary_Record]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[P_GECS_Determine_Primary_Record]
GO

SET QUOTED_IDENTIFIER ON 
GO
SET ANSI_NULLS ON 
GO

Create Proc P_GECS_Determine_Primary_Record
As
/*********************************************************************/
/*		GEAC - Determine Primary Record			     */
/*                  						     */
/* Purpose: Determine Primary Record for GEAC		             */
/*                                                                   */
/*  Org. Date: 03/20/2009        EPAK                                */
/*  Mod. Date: 00/00/0000        XXXX                                */
/*                                                                   */
/*********************************************************************/

SET NOCOUNT ON
/*********************************************************************/
/*								     */
/*********************************************************************/
Declare		@ErrorCode		Int,
		@Process		Varchar(256),
		@GID			Char(12),
		@OptID			Char(9),
		@HireDate		DateTime,
		@TermDate		Datetime,
		@iGID			Char(12),
		@iOptID			Char(9),
		@iTermDate		Datetime,
		@iHireDate		DateTime,
		@fGID			Char(12),
		@fOptID			Char(9),
		@fHireDate		DateTime,
		@fTermDate		Datetime,
		@P			Char(1),
		@S			Char(1),
		@ERR			Char(3),
		@SeqNo			Int,
		@One			Int,
		@Max			Int,
		@Zero			Int,
		@iSeqNo			Int,
		@iMax			Int

/*********************************************************************/
/*								     */
/*********************************************************************/
SET	@ERR		= 'ERR'
SET	@P		= 'P'
SET	@S		= 'S'
SET	@SeqNo		= 1
SET	@One		= 1
SET	@Zero		= 0
SET	@Process	= 'P_GEAC_Determine_Primary_Record'

/*********************************************************************/
/*								     */
/*********************************************************************/
Truncate Table TempGECSDups

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Unable to Truncate TempGECSDups Table FAILURE')
		RAISERROR (70007,16,1)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Delete	WYETHaux.dbo.tblGecsEmployee 
Where 	TERM_REASON_CODE = @ERR

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','FAILURE to Delete ERR Record(s) from WYETHaux.dbo.tblGecsEmployee Table')
		RAISERROR (70007,16,1)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Create	Table	#Determine
 (
	GID		Char(12)	NOT NULL,
	OptID		Char(10)	NOT NULL,
	Hiredate	Datetime	NULL,
	TermDate	Datetime	NULL
 )

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Create #Determine Table FAILURE')
		RAISERROR (70007,16,2)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Create	Table	#iDetermine
 (
	SeqNo		Int		Identity,
	GID		Char(12)	NOT NULL,
	OptID		Char(9)		NOT NULL,
	Hiredate	Datetime	NULL,
	TermDate	Datetime	NULL
 )

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Create #iDetermine Table FAILURE')
		RAISERROR (70007,16,2)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Create	Table	#Info
 (
	SeqNo		Int		Identity,
	OptID		Char(9)		NOT NULL
 )

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Create #Info Table FAILURE')
		RAISERROR (70007,16,2)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Insert	#Info
 (
	OptID
 )

Select	DISTINCT
	Substring(OPT_ID,2,9)

From	tblGECSemployee
Where	Substring(OPT_ID,2,9) IN 
 (
	Select	Substring(OPT_ID,2,9)
	From	tblGECSemployee
	Group By Substring(OPT_ID,2,9)
	Having COUNT(Substring(OPT_ID,2,9)) > @One
 )
Order By Substring(OPT_ID,2,9)

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Insert INTO #Info Table FAILURE')
		RAISERROR (70007,16,3)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Insert	#Determine
 (
	GID,OptID,HireDate,TermDate
 )

Select	GLOBAL_ID,
	Substring(OPT_ID,2,9),
	CURRENT_HIRE_DATE,
	TERM_DATE

From	tblGECSemployee
Where	Substring(OPT_ID,2,9) IN
 (
	Select	Substring(OPT_ID,2,9)
	From	tblGECSemployee
	Group By Substring(OPT_ID,2,9)
	Having COUNT(Substring(OPT_ID,2,9)) > @One
 )

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Insert INTO #Determine Table FAILURE')
		RAISERROR (70007,16,3)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Begin Tran GEAC

/*********************************************************************/
/*		Update NON Duplicate Record(s)			     */
/*********************************************************************/
Update	tblGECSemployee
SET	Primary_Record	= @P
From	tblGECSemployee
Where	Substring(OPT_ID,2,9) NOT IN
 (
	Select	OptID From #Info
 )

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Rollback Tran GEAC
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Update tblGECSemployee Table FAILURE')
		RAISERROR (70007,16,4)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*		Determine The Count				     */
/*********************************************************************/
Select	@Max	= COUNT(SeqNo)
From	#Info

/*********************************************************************/
/*								     */
/*********************************************************************/
WHILE @SeqNo <= @Max
 Begin	---- BEGIN OUTER LOOP
	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Truncate Table	#iDetermine

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	SET	@iMax		= @Zero
	SET	@iSeqNo		= @One

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Select	@OptID	= OptID
	From	#Info
	Where	SeqNo	= @SeqNo

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Insert	#iDetermine
	 (
		GID,OptID,Hiredate,TermDate
	 )
	Select	GID,OptID,Hiredate,TermDate
	From	#Determine
	Where	OptID = @OptID
	Order By OptID,
	 Case When TermDate IS NULL Then 'A' Else 'T' End,
	 Case When TermDate IS NULL Then HireDate Else TermDate End Desc

		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			Rollback Tran GEAC
			Insert	ProcessLog
			Values(Getdate(),@Process,@ErrorCode,'FAILURE','Insert INTO #iDetermine Table FAILURE')
			RAISERROR (70007,16,5)
			RETURN @ErrorCode
		 End

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Select	@iMax	= COUNT(GID)
	From	#iDetermine

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	WHILE @iSeqNo <= @iMax
	 Begin	---- BEGIN INNER LOOP

		/*********************************************************/
		/*					                 */
		/*********************************************************/
		Select	@GID		= GID,
			@OptID		= OptID,
			@HireDate	= HireDate,
			@TermDate	= TermDate
		From	#iDetermine
		Where	SeqNo		= @iSeqNo

		/*********************************************************/
		/*					                 */
		/*********************************************************/
		IF @iSeqNo = @One
		 Begin
			/*************************************************/
			/*				                 */
			/*************************************************/
			Select	@fGID		= @GID,
				@fOptID		= @OptID,
				@fHireDate	= @HireDate,
				@fTermDate	= @TermDate
		
			Select	@iGID		= @GID,
				@iOptID		= @OptID,
				@iHireDate	= @HireDate,
				@iTermDate	= @TermDate

			/*************************************************/
			/*				                 */
			/*************************************************/
			SET ROWCOUNT 1

			Update	tblGECSemployee
			Set	Primary_Record		= @P
			Where	GLOBAL_ID 		= @fGID
			And	Substring(OPT_ID,2,9)	= @fOptID
			And	CURRENT_HIRE_DATE	= @fHireDate
			And	ISNULL(TERM_DATE,1)	= ISNULL(@fTermDate,1)

			SET ROWCOUNT 0


				Select	@ErrorCode = @@ERROR
				IF @ErrorCode <> @Zero
				 Begin
					Rollback Tran GEAC
					Insert	ProcessLog
					Values(Getdate(),@Process,@ErrorCode,'FAILURE','Update tblGECSemployee Table FAILURE')
					RAISERROR (70007,16,6)
					RETURN @ErrorCode
				 End

			/*************************************************/
			/*				                 */
			/*************************************************/	
		 End
		ELSE
		 Begin
			/*************************************************/
			/*				                 */
			/*************************************************/
			SET @iSeqNo = @iMax

			/*************************************************/
			/*				                 */
			/*************************************************/
		 End

		/*********************************************************/
		/*					                 */
		/*********************************************************/
		Select	@iSeqNo = @iSeqNo + @One

		/*********************************************************/
		/*					                 */
		/*********************************************************/
	 End	---- END INNER LOOP

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Select	@SeqNo = @SeqNo + @One

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
 End	---- END OUTER LOOP

/*********************************************************************/
/*								     */
/*********************************************************************/
DROP TABLE	#Info
DROP TABLE	#iDetermine
DROP TABLE	#Determine


/*********************************************************************/
/*								     */
/*********************************************************************/
Insert	TempGECSDups
Select	GLOBAL_ID,
	OPT_ID,
	FIRST_NAME,
	LAST_NAME,
	COUNTRY,
	AFFILIATE_CODE,
	AFFILIATE_DESCRIPTION


From	tblGECSemployee
Where	Substring(OPT_ID,2,9) IN
(
	Select	Substring(OPT_ID,2,9)
	From	tblGECSemployee
	Where	Substring(OPT_ID,2,9) IN
	 (
		Select	Substring(OPT_ID,2,9)
		From	tblGECSemployee
		Group By Substring(OPT_ID,2,9)
		Having COUNT(Substring(OPT_ID,2,9)) > 1
	 )
	And	TERM_DATE IS NULL
	And	Primary_Record = @S
 )
Order By Substring(OPT_ID,2,9),Primary_Record

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Rollback Tran GEAC
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Insert TempGEACDups Table FAILURE')
		RAISERROR (70007,16,7)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Delete	tblGECSemployee
Where	Primary_Record = @S

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		Rollback Tran GEAC
		Insert	ProcessLog
		Values(Getdate(),@Process,@ErrorCode,'FAILURE','Delete Non Primary record(S) From tblGECSemployee Table FAILURE')
		RAISERROR (70007,16,8)
		RETURN @ErrorCode
	 End

/*********************************************************************/
/*								     */
/*********************************************************************/
Commit Tran GEAC

/*********************************************************************/
/*								     */
/*********************************************************************/
SET NOCOUNT OFF

GO
SET QUOTED_IDENTIFIER OFF 
GO
SET ANSI_NULLS ON 
GO

